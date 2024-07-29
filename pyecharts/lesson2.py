from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.faker import Faker

bar = Bar()
bar.add_xaxis(Faker.choose())
bar.add_yaxis("a", Faker.values(), color=Faker.rand_color())
bar.add_yaxis("b", Faker.values(), color=Faker.rand_color())
bar.set_series_opts(markline_opts=opts.MarkLineOpts(
    data=[opts.MarkLineItem(y=80, name="合格指标")]
))
bar.set_global_opts(title_opts=opts.TitleOpts(title="直方图"))

bar.render("随机选择.html")