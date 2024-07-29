from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType

bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
bar.add_xaxis(["语文", "数学", "英语", "理综"])
bar.add_yaxis("成绩", [122, 115, 137, 214])
bar.set_global_opts(title_opts=opts.TitleOpts(title="期末考试", subtitle="小樊"))
bar.render("成绩表.html")