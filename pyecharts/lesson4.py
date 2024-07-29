from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.faker import Faker

bar = Bar()
bar.add_xaxis(Faker.choose())
bar.add_yaxis("a", Faker.values(), stack='stack', color=Faker.rand_color())
bar.add_yaxis("b", Faker.values(), stack='stack', color=Faker.rand_color())
bar.set_global_opts(title_opts=opts.TitleOpts(title="堆叠图"))
bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))

bar.render("堆叠图.html")