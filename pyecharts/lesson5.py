from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.faker import Faker

pie = Pie()
pie.add('', [list(z) for z in zip(Faker.choose(), Faker.values())])
pie.set_global_opts(title_opts=opts.TitleOpts(title="Pie图表"))

pie.render("饼图.html")