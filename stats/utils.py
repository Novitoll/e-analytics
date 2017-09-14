import os
import calendar
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from datetime import timedelta

MEGABYTE = 1048576  # 1024*1024


def parse_datetime_with_timezone(date_str):
    try:
        timezone = date_str[-5:]
        dt = datetime.datetime.strptime(date_str[:-5], '%d/%b/%Y:%H:%M:%S')
    except Exception, ex:
        print "[-] Can not convert string to datetime {}".format(ex)
        return None

    hours = int(timezone[1:3])

    # normalize timezone to GMT 0000
    if timezone[0] == "+":
        dt += timedelta(hours=hours)
        return dt
    elif timezone[0] == "-":
        dt -= timedelta(hours=hours)
        return dt
    else:
        return None


def ip_to_int32(ip):
    int32 = None
    try:
        int32 = reduce(lambda a, b: a << 8 | b, map(int, ip.split(".")))
    except Exception, ex:
        print "[-] Can not convert to int32 with {0}, error: \n{1}".format(ip, ex)
    return int32


def get_office_data_from_ip(ip, offices_conf=None):
    int32 = ip_to_int32(ip)
    office_data = dict(office='external', country='unknown', region='unknown', retranslator_name='unknown')

    if int32 is None:
        return office_data
    for host_ip_min, host_data in offices_conf.iteritems():
        if int(host_ip_min) <= int32 <= int(host_data['ip_int_max']):
            office_data = host_data
            break
    return office_data


def get_weekday(x):
    try:
        return calendar.day_name[x.weekday()]
    except Exception, ex:
        print "[-] Can not get weekday from {0} datetime, error: \n {1}".format(x, ex)
        return None


def get_hour(x, offset):
    try:
        x += timedelta(hours=offset)
        return x.hour
    except Exception, ex:
        print "[-] Can not add timedelta for {0} datetime, error: \n {1}".format(x, ex)
        return None


def convert2mega(x, bit=False):
    # transform Bytes to Mbits
    return x * 8 / MEGABYTE if bit else x / MEGABYTE


def to_excel(df, sheet_name, writer, **kwargs):
    df.to_excel(writer, sheet_name=sheet_name, **kwargs)
    writer.save()


def X_vs_unique_count_barchart(x, y, df, out=os.getcwd()):
    x_label = x
    y_label = '%s unique count' % y
    title = "{0} distr per {1}".format(y, x)

    df_bar = df.groupby([x])[y].nunique()
    ax = df_bar.plot(kind='bar', title=title)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.tick_params(labelsize=16)
    plt.xticks(rotation=90)
    ax.figure.savefig(os.path.join(out, '%s_barplot.png' % x_label))


def X_vs_count_multiclass_barchart(x, y, c, ddf, out=os.getcwd()):
    x_label = x
    y_label = '%s count' % y

    df = ddf.groupby([x, c])[y].count().reset_index()

    ax = sns.factorplot(data=df, x=x, y=y, hue=c, kind="bar", size=4, aspect=3)
    ax.set_xlabels(x_label)
    ax.set_ylabels(y_label)
    ax.set_xticklabels(rotation=45)
    ax.fig.suptitle("{0} distr per {1} in {2} classes".format(y, x, c))
    ax.figure.savefig(os.path.join(out, '%s_factorplot.png' % x_label))


def X_distplot(x, df, x_label=None, out=os.getcwd()):
    if x_label:
        x_label = x
    ax = sns.distplot(df[x])
    ax.set(xlabel=x_label, title="%s distr with PDF" % x)
    ax.tick_params(labelsize=16)
    plt.xticks(rotation=90)
    ax.figure.savefig(os.path.join(out, '%s_distplot.png' % x_label))


def X_countplot_distr(x_label, df, order, out=os.getcwd()):
    plt.figure(figsize=(12, 8))
    ncount = len(df)
    ax = sns.countplot(x=x_label, data=df, order=order)
    plt.title('Distribution of %s' % x_label)
    plt.xlabel(x_label)

    # Make twin axis
    # ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    # ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    # ax2.yaxis.set_label_position('left')

    # ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                    ha='center', va='bottom')  # set the alignment of the text

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))
    ax.figure.savefig(os.path.join(out, '{}_countplot_distribution.png'.format(x_label)))
