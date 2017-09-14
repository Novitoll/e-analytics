import os
import time
import sys
import re
import cPickle
import argparse
import datetime
import pandas as pd
import numpy as np

# columns
USER_L = 'User\'s Login'
CHANNEL_SLUG = 'Channel Slug'
SLUG = 'Slug'
ACTION = 'Action'
DURATION = 'Duration in sec'
OFFICE = 'Office (City)'
REF = 'Referrer'
USER_DT = 'User\'s Datetime (UTC+3)'

col_views_c = 'view count'
col_watching_c = 'watching count'
col_watching_dur_sum = 'duration sum in min'
dt_format = '%Y-%m-%d %H:%M:%S'


def main(args):
    t0 = time.time()
    itw_slugs = [x.strip() for x in args.slugs.split(', ')]
    pkl_file = os.path.join(args.out, 'df.pkl')

    if not os.path.isfile(pkl_file):
        print "[ ] Loading data.."
        piwik_data_path = args.data
        columns = ["User's Datetime (UTC+3)", "Office (City)",
                   "User's Login", "Internal IP", "Content Type",
                   "Slug", "Action", "Retranslator", "Channel Slug", "Referrer",
                   "Country", "Duration in sec"]

        if str(args.data).endswith('xlsx'):
            df = pd.read_excel(piwik_data_path, convert_float=False, usecols=columns)
        elif str(args.data).endswith('csv'):
            df = pd.read_csv(piwik_data_path, convert_float=False, usecols=columns)
        else:
            raise Exception('unknown input data format. Use either xls* or csv')

        print "[+] User action stats from Piwik has been loaded. Shape is {}".format(df.shape)

        # Pre-processing
        df.drop(df[(df[SLUG].isnull()) | (~df[ACTION].isin(['view', 'stop', 'play', 'watching']))].index, inplace=True)

        if len(itw_slugs):
            print "[ ] Filtering by given slugs.."
            df.drop(df[~df[SLUG].isin(itw_slugs)].index, inplace=True)

        # convert old format milliseconds to minutes
        df[DURATION] = df[DURATION].apply(lambda x: x / (1024 * 60))

        print '[ ] Shape after filtering -- {}'.format(df.shape)
        print '[ ] Unique users by {0} -- {1}'.format(USER_L, len(df[USER_L].unique()))
        print '[ ] Registered number of unique channel slugs -- {}'.format(len(df[CHANNEL_SLUG].unique()))
        print '[ ] Registered number of unique slugs -- {}'.format(len(df[SLUG].unique()))

        if args.parse_date:
            print "[ ] Parsing dates.."

            def _parse_date(x):
                #     2017-04-04 18:54:36
                try:
                    return datetime.datetime.strptime(str(x), dt_format)
                except Exception, ex:
                    print "[-] Error {}".format(ex)
                    return None

            df[USER_DT] = df[USER_DT].apply(lambda x: _parse_date(x))
            start = datetime.datetime.strptime(args.start, dt_format)
            end = datetime.datetime.strptime(args.end, dt_format)

            df = df.loc[(df[USER_DT] >= start) & (df[USER_DT] <= end)]

        # to pickle
        print "[ ] Storing processed dataframe to %s pickle.." % pkl_file
        with open(pkl_file, 'wb+') as f:
            cPickle.dump(df, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
    else:
        print "[+] Pickle was found. Loading pickle.."
        with open(pkl_file, 'rb+') as f:
            df = cPickle.load(f)
            f.close()

    # Generating reports
    # 1. generate ITW_reports.xlsx
    print "[ ] Generating overall data report.."
    un_w, un_o = do_all_stream_stats(df, args)

    # 2. generate excel report per slug
    print "[ ] Generating report per slug.."
    for slug in df[SLUG].unique():
        df_loc = df.loc[df[SLUG] == slug]
        if df_loc is None:
            print "[-] {} is not in df dataframe".format(slug)
            continue
        do_stats_per_stream(df_loc, args, slug=slug)

    # 3. summary
    print "\n[ ] {0}: {1} unique watchers with > 0 sec watching duration time from {2} unique offices for {3} streams."\
        .format(args.date, un_w, un_o, len(itw_slugs))

    print "[+] Piwik analytics has been completed in %.3f sec" % (time.time() - t0)


def get_hostname_from_ref(ref):
    try:
        a = re.sub(r"http.*//", "", ref)
        return re.sub(r"/.*", "", a)
    except Exception, ex:
        print "[-] Could not process referrer - {0}, error: \n{1}".format(ref, ex)
        return None


def do_view_watch_stats(df):
    irr_cols = [0, 1, 3, 4, 6, 8, 9, 10, 11]  # irrelevant columns to drop
    # prepare dataframes for Viewers and Watchers
    views_df = df.loc[df[ACTION] == "view"].drop(df.columns[irr_cols], axis=1)
    watching_df = df.loc[df[ACTION] == "watching"].drop(df.columns[irr_cols[:-1]], axis=1)
    location_df = df.drop(df.columns[[0, 2, 3, 4] + range(6, 11)], axis=1)

    # viewers
    unique_viewers = pd.Series(df.loc[df[ACTION] == "view"][USER_L].unique()).to_frame(name="unique viewers")
    unique_offices = pd.Series(df.loc[df[ACTION] == "view"][OFFICE].unique()).to_frame(name="unique offices")
    unique_v_stats = pd.concat([unique_viewers, unique_offices], axis=1)

    # watchers
    unique_watchers = pd.Series(df.loc[df[ACTION] == "watching"][USER_L].unique()).to_frame(name="unique watchers")
    unique_offices = pd.Series(df.loc[df[ACTION] == "watching"][OFFICE].unique()).to_frame(name="unique offices")
    unique_w_stats = pd.concat([unique_watchers, unique_offices], axis=1)

    # referrers
    filtered_ref_df = df.drop(df[df[REF].isnull()].index)
    filtered_ref_df.loc[:, REF] = filtered_ref_df[REF].apply(lambda x: get_hostname_from_ref(x))
    view_referrer_freq_df = filtered_ref_df.loc[filtered_ref_df[ACTION] == "view"][REF] \
        .value_counts().to_frame(name='viewer referrer counts')
    watch_referrer_freq_df = filtered_ref_df.loc[filtered_ref_df[ACTION] == "watching"][REF] \
        .value_counts().to_frame(name='watcher referrer counts')

    return views_df, watching_df, location_df, unique_v_stats, unique_w_stats, view_referrer_freq_df, watch_referrer_freq_df


def do_stats_by_watching_duration(watching_df, df):
    # Total unique watchers with the list of User's login and total time of watching (summarize for all streams)
    # and number of streams (which user have watched) - for potential engagement
    watching_df_dur = watching_df.drop(watching_df[watching_df[DURATION].isnull()].index)

    # 1.
    try:
        total_dur_by_watching_user = watching_df_dur.groupby([USER_L])[DURATION] \
            .sum() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={DURATION: 'Total time of duration in minutes for all watched streams'})
        median_dur_by_watching_user = watching_df_dur.groupby([USER_L])[DURATION] \
            .median() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={DURATION: 'Median time of duration in minutes for all watched streams'})
        mean_dur_by_watching_user = watching_df_dur.groupby([USER_L])[DURATION] \
            .mean() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={DURATION: 'Average time of duration in minutes for all watched streams'})
        total_watched_stream_per_user = watching_df_dur.groupby([USER_L])[SLUG] \
            .nunique() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={SLUG: 'count of unique watched streams'})

        merged = pd.merge(total_dur_by_watching_user, total_watched_stream_per_user, on=USER_L)
        merged2 = pd.merge(median_dur_by_watching_user, mean_dur_by_watching_user, on=USER_L)
        all_merged = pd.merge(merged, merged2, on=USER_L)
    except Exception, ex:
        raise Exception("Error occurred in do_stats_by_watching_duration#1\n{}".format(ex))

    # 2. condition - location (office)
    try:
        location_df_cond = df.drop(df.columns[[0, 2, 3, 4] + range(6, 10)], axis=1)  # include Duration at(10)
        location_df_dur = location_df_cond.drop(location_df_cond[location_df_cond[DURATION].isnull()].index)
        location_df_dur.loc[:, DURATION] = location_df_dur[DURATION].apply(lambda x: x / 60)

        total_dur_by_watching_office = location_df_dur.groupby([OFFICE])[DURATION] \
            .sum() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={DURATION: 'Total duration in minutes for all watched streams'})
        total_watched_stream_per_office = location_df_dur.groupby([OFFICE])[SLUG] \
            .nunique() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={SLUG: 'Count of watched streams'})
        mean_dur_by_watching_office = location_df_dur.groupby([OFFICE])[DURATION] \
            .mean() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={DURATION: 'Average duration in minutes for all watched streams'})
        median_dur_by_watching_office = location_df_dur.groupby([OFFICE])[DURATION] \
            .median() \
            .sort_values(ascending=False) \
            .reset_index() \
            .rename(columns={DURATION: 'Median duration in minutes for all watched streams'})

        merged_o = pd.merge(total_dur_by_watching_office, total_watched_stream_per_office, on=OFFICE)
        merged2_o = pd.merge(mean_dur_by_watching_office, median_dur_by_watching_office, on=OFFICE)
        all_merged_o = pd.merge(merged_o, merged2_o, on=OFFICE)
    except Exception, ex:
        raise Exception("Error occurred in do_stats_by_watching_duration#2\n{}".format(ex))

    return all_merged, all_merged_o


def do_all_stream_stats(df, args):
    reports_path = os.path.join(args.out, '_general.xlsx')
    writer = pd.ExcelWriter(reports_path)

    # 1. Condition - per stream
    views_df, watching_df, location_df, \
    unique_v_stats, unique_w_stats, view_referrer_freq_df, watch_referrer_freq_df = do_view_watch_stats(df)
    try:
        all_merged, all_merged_o = do_stats_by_watching_duration(watching_df, df)
    except Exception, ex:
        print "[-] {}".format(ex)

    views = pd.pivot_table(views_df, values=ACTION, index=[SLUG, USER_L], aggfunc='count') \
        .to_frame(name=col_views_c)
    watching = pd.pivot_table(watching_df, values=ACTION, index=[SLUG, USER_L], aggfunc='count') \
        .to_frame(name=col_watching_c)
    watching_dur = pd.pivot_table(watching_df, values=DURATION, index=[SLUG, USER_L], aggfunc=np.sum) \
        .to_frame(name=col_watching_dur_sum)

    # condition - location (office)
    office_g = df.loc[df[ACTION] == "watching"].groupby(SLUG)[OFFICE] \
        .value_counts() \
        .unstack().stack(dropna=False) \
        .reset_index(name="watching count") \
        .set_index([SLUG, OFFICE])

    # 2. Condition - for all streams

    # all: condition - view count
    views = views_df.groupby([SLUG])[ACTION] \
        .count() \
        .sort_values(ascending=False) \
        .reset_index() \
        .rename(columns={ACTION: "count of 'view' action"})

    # all: condition - watch count
    watching = watching_df.groupby([SLUG])[ACTION] \
        .count() \
        .sort_values(ascending=False) \
        .reset_index() \
        .rename(columns={ACTION: "count of watched streams"})

    # writing
    all_merged.to_excel(writer, 'allwatchedusersstats')
    all_merged_o.to_excel(writer, 'allwatchedofficesstats')
    views.to_excel(writer, 'allstreams-viewcount')
    watching.to_excel(writer, 'allstreams-watchcount')

    watching_dur.to_excel(writer, 'watchdurationsum')

    unique_v_stats.to_excel(writer, 'uniqueviewerstats')
    unique_w_stats.to_excel(writer, 'uniquewatcherstats')
    view_referrer_freq_df.to_excel(writer, 'referrerviewstats')
    watch_referrer_freq_df.to_excel(writer, 'referrerwatchstats')

    # TODO: merge these
    views.to_excel(writer, 'viewsusercount')
    watching.to_excel(writer, 'watchusercount')
    office_g.to_excel(writer, 'watchofficecount')

    writer.save()

    print "[+] Done"

    # for summary
    uniquewatchers = len(all_merged[USER_L].unique())
    uniqueoffices = len(all_merged_o.loc[all_merged_o[OFFICE] != 'Out of Range'][OFFICE].unique())

    return uniquewatchers, uniqueoffices


def do_stats_per_stream(df, args, slug):
    writer = pd.ExcelWriter(os.path.join(args.out, '%s.xlsx' % slug))

    views_df, watching_df, location_df, \
    unique_v_stats, unique_w_stats, view_referrer_freq_df, watch_referrer_freq_df = do_view_watch_stats(df)

    watching_office_g = df.loc[df[ACTION] == "watching"][OFFICE].value_counts().to_frame('office watch count')

    views = views_df.groupby([USER_L])[ACTION].count() \
        .reset_index(name=col_views_c)
    watching = watching_df.groupby([USER_L])[ACTION].count() \
        .reset_index(name=col_watching_c)

    watching_dur = watching_df.groupby([USER_L])[DURATION].sum() \
        .reset_index(name=col_watching_dur_sum)
    mean = watching_dur[col_watching_dur_sum].mean()
    median = watching_dur[col_watching_dur_sum].median()

    watching_dur_stats_df = pd.DataFrame(data=[['average of duration sum in min', mean],
                                               ['median of duration sum in min', median]])

    # writing
    watching_dur_stats_df.to_excel(writer, 'watchdurationstats')
    watching_dur.to_excel(writer, 'watchdurationsum')
    unique_v_stats.to_excel(writer, 'uniqueviewerstats')
    unique_w_stats.to_excel(writer, 'uniquewatcherstats')
    view_referrer_freq_df.to_excel(writer, 'referrerviewstats')
    watch_referrer_freq_df.to_excel(writer, 'referrerwatchstats')
    watching_office_g.to_excel(writer, 'watchofficecount')
    views.to_excel(writer, 'viewscount')
    watching.to_excel(writer, 'watchcount')

    writer.save()


def main_general(args):
    t0 = time.time()
    root_dir = args.data
    pkl_file = os.path.join(args.out, 'df_general.pkl')
    try:
        dates = [x.strip() for x in args.dates.split(', ')]
    except Exception, ex:
        print "[-] {}".format(ex)
        return

    dfs = []
    for d in dates:
        pkl = os.path.join(root_dir, d, 'df.pkl')
        print "[ ] Loading %s.." % pkl
        with open(pkl, 'rb') as f:
            df = cPickle.load(f)
            dfs.append(df)
            f.close()

    print "[ ] Concatenating dataframes into one.."
    all_days_df = pd.concat(dfs)
    print "[ ]  Shape is {}".format(all_days_df.shape)

    print "[ ] Storing processed dataframe to %s pickle.." % pkl_file
    with open(pkl_file, 'wb+') as f:
        cPickle.dump(all_days_df, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    print "[ ] Generating overall data report.."
    un_w, un_o = do_all_stream_stats(all_days_df, args)
    print "[+] Done. %.3f sec" % (time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating excel reports for stream stats.')
    parser.add_argument('--date', help='Date')
    parser.add_argument('--data', help='Input filepath to Piwik data')
    parser.add_argument('--out', help='Output directory for reports', default=os.getcwd())
    parser.add_argument('--slugs', help='String of stream slugs divided by comma', default=[])
    parser.add_argument('--parse_date', help='Boolean flag if date need to be parsed', default=False, type=bool)
    parser.add_argument('--start', help='Start of datetime in Y-m-d H:M:S format')
    parser.add_argument('--end', help='End of datetime in Y-m-d H:M:S format', default=[])
    parser.add_argument('--dates', help='String of dates', default="")
    parser.add_argument('--general', help='Is general report?', default=False, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if not os.path.isdir(args.out):
        try:
            os.mkdir(args.out)
        except Exception, ex:
            print "[-] Could not create {0} dir, error: {1}".format(args.out, ex)

    if args.general:
        main_general(args)
    else:
        main(args)
