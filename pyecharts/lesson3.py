from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.faker import Faker

bar = Bar()
bar.add_xaxis(Faker.choose())
bar.add_yaxis("a", Faker.values(), color=Faker.rand_color())
bar.add_yaxis("b", Faker.values(), color=Faker.rand_color())
bar.reversal_axis()
bar.set_series_opts(label_opts=opts.LabelOpts(position="right"))
bar.set_global_opts(title_opts=opts.TitleOpts(title="XY反转"))

bar.render("翻转图.html")