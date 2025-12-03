import json
import os
from datetime import datetime


class EChartsOptimizedVisualizer:
    def __init__(self):
        """初始化ECharts优化可视化器"""
        self.data_files = [
            'ECharts_桑基图数据.json',
            'ECharts_雷达图数据.json',
            'ECharts_动画时间序列数据.json',
            'ECharts_3D表面图数据.json'
        ]

        print(" ECharts 优化可视化分析器已初始化")
        print("专注于4个最适合ECharts的图表类型")
        self.check_data_files()

    def check_data_files(self):
        """检查数据文件是否存在"""
        missing_files = []
        for file in self.data_files:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            print("  缺少以下数据文件:")
            for file in missing_files:
                print(f"   - {file}")
            print("请先运行 Echarts_数据准备.py 生成数据文件")
        else:
            print(" 所有数据文件检查完成")

    def load_data_file(self, filename):
        """加载JSON数据文件"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f" 文件 {filename} 未找到")
            return None
        except json.JSONDecodeError:
            print(f" 文件 {filename} JSON格式错误")
            return None

    def create_html_template(self):
        """创建HTML基础模板"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>全球碳排放量数据可视化分析</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.0/dist/echarts.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Microsoft YaHei', 'PingFang SC', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            padding: 30px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}

        .header h1 {{
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header p {{
            font-size: 1.2em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}

        .meta-info {{
            font-size: 0.9em;
            color: #95a5a6;
        }}

        .chart-grid {{
            display: flex;
            flex-direction: column;
            gap: 25px;
            margin-bottom: 30px;
        }}

        .chart-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .chart-container:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }}

        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}

        .chart-description {{
            font-size: 0.95em;
            color: #666;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}

        .chart-title::before {{
            content: "📊";
            margin-right: 10px;
            font-size: 1.2em;
        }}

        .chart {{
            width: 100%;
            height: 400px;
            margin: 0 auto;
        }}

        .chart.large {{
            height: 500px;
        }}

        .chart.small {{
            height: 350px;
        }}

        .full-width {{
            width: 100%;
        }}

        /* 让雷达图和3D图在容器中居中 */
        .chart-container .chart {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 400px;
            color: #7f8c8d;
            font-size: 1.1em;
        }}

        .error {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 400px;
            color: #e74c3c;
            font-size: 1.1em;
            background: #fdf2f2;
            border-radius: 10px;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2em;
            }}

            .chart {{
                height: 300px;
            }}

            .chart-container {{
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌍 全球碳排放量数据可视化分析</h1>
            <p>基于联合国气候变化框架公约(UNFCCC)官方数据</p>
            <div class="meta-info">
                数据时间范围: 1990-2020年 | 覆盖国家: 44个 | 生成时间: {generation_time}
            </div>
        </div>

        <div class="chart-grid">
            {chart_containers}
        </div>

        <div class="footer">
            <p>💡 数据来源: UNFCCC GHG Data Interface | 可视化工具: ECharts 5.4.0</p>
            <p>📊 图表支持缩放、筛选、数据导出等交互功能</p>
        </div>
    </div>

    <script>
        // 全局配置
        const globalConfig = {{
            theme: 'light',
            backgroundColor: 'transparent',
            textStyle: {{
                fontFamily: 'Microsoft YaHei, PingFang SC, Arial, sans-serif'
            }}
        }};

        // 工具函数
        function formatNumber(num) {{
            if (num >= 1e6) {{
                return (num / 1e6).toFixed(1) + 'M';
            }} else if (num >= 1e3) {{
                return (num / 1e3).toFixed(1) + 'K';
            }}
            return num.toFixed(0);
        }}

        function createTooltipFormatter(unit = 'kt CO₂当量') {{
            return function(params) {{
                if (Array.isArray(params)) {{
                    let result = params[0].axisValue + '<br/>';
                    params.forEach(param => {{
                        result += param.marker + param.seriesName + ': ' + 
                                 formatNumber(param.value) + ' ' + unit + '<br/>';
                    }});
                    return result;
                }} else {{
                    return params.name + '<br/>' + 
                           params.marker + params.seriesName + ': ' + 
                           formatNumber(params.value) + ' ' + unit;
                }}
            }};
        }}

        // 响应式处理
        function makeResponsive() {{
            const charts = document.querySelectorAll('[id^="chart"]');
            charts.forEach(chartElement => {{
                const chartInstance = echarts.getInstanceByDom(chartElement);
                if (chartInstance) {{
                    chartInstance.resize();
                }}
            }});
        }}

        window.addEventListener('resize', makeResponsive);

        // 图表初始化
        {chart_scripts}
    </script>
</body>
</html>'''

    def create_timeseries_chart(self):
        print("    创建动画时间序列图...")

        data = self.load_data_file('ECharts_动画时间序列数据.json')
        if not data:
            return None

        container = '''
            <div class="chart-container">
                <div class="chart-title">📈 动画时间序列：主要排放国时间趋势分析</div>
                <p class="chart-description">
                    <strong>ECharts优势：</strong>极其流畅的动画效果，支持时间轴播放和交互控制，数据点动态出现消失
                </p>
                <div id="chartTimeseries" class="chart large"></div>
            </div>'''

        script = f'''
        // 时间序列图表
        (function() {{
            const data = {json.dumps(data, ensure_ascii=False)};
            const chart = echarts.init(document.getElementById('chartTimeseries'));

            const option = {{
                title: {{
                    text: '1990-2020年主要排放国排放量变化趋势',
                    left: 'center',
                    textStyle: {{ fontSize: 16, fontWeight: 'bold' }}
                }},
                toolbox: {{
                    feature: {{
                        saveAsImage: {{
                            name: '主要排放国时间趋势',
                            type: 'png',
                            backgroundColor: '#fff'
                        }}
                    }}
                }},
                tooltip: {{
                    trigger: 'axis',
                    formatter: createTooltipFormatter(),
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: '#ccc',
                    textStyle: {{ color: '#333' }}
                }},
                legend: {{
                    data: data.series.map(s => s.name),
                    top: 40,
                    type: 'scroll'
                }},
                grid: {{
                    left: '3%',
                    right: '4%',
                    bottom: '8%',
                    top: '15%',
                    containLabel: true
                }},
                xAxis: {{
                    type: 'category',
                    data: data.years,
                    axisLabel: {{ rotate: 45 }}
                }},
                yAxis: {{
                    type: 'value',
                    name: '排放量 (kt CO₂当量)',
                    axisLabel: {{
                        formatter: function(value) {{
                            return formatNumber(value);
                        }}
                    }}
                }},
                dataZoom: [
                    {{
                        type: 'inside',
                        xAxisIndex: 0,
                        start: 0,
                        end: 100
                    }},
                    {{
                        type: 'slider',
                        xAxisIndex: 0,
                        start: 0,
                        end: 100,
                        bottom: 20
                    }}
                ],
                series: data.series.map(s => ({{
                    name: s.name,
                    type: 'line',
                    data: s.data,
                    symbolSize: 6,
                    lineStyle: {{ width: 3 }},
                    emphasis: {{
                        focus: 'series',
                        lineStyle: {{ width: 4 }}
                    }},
                    animationDelay: function (idx) {{
                        return idx * 100;
                    }}
                }}))
            }};

            chart.setOption(option);
        }})();'''

        return container, script

    def create_radar_chart(self):
        print("    创建雷达图...")

        data = self.load_data_file('ECharts_雷达图数据.json')
        if not data:
            return None

        container = '''
            <div class="chart-container">
                <div class="chart-title">📡 雷达图：主要排放国多维对比</div>
                <p class="chart-description">
                    <strong>ECharts优势：</strong>极强的交互性（悬停、选择、缩放），支持多系列对比，流畅自然的动画效果
                </p>
                <div id="chartRadar" class="chart"></div>
            </div>'''

        script = f'''
        // 雷达图
        (function() {{
            const data = {json.dumps(data, ensure_ascii=False)};
            const chart = echarts.init(document.getElementById('chartRadar'));

            const option = {{
                title: {{
                    text: 'Top 6 排放国多维指标对比',
                    left: 'center'
                }},
                toolbox: {{
                    feature: {{
                        saveAsImage: {{
                            name: '排放国多维指标雷达图',
                            type: 'png',
                            backgroundColor: '#fff'
                        }}
                    }}
                }},
                tooltip: {{
                    trigger: 'item',
                    formatter: function(params) {{
                        let result = params.name + '<br/>';
                        params.value.forEach((value, index) => {{
                            const indicator = data.indicators[index];
                            result += indicator.name + ': ' + formatNumber(value);
                            if (indicator.name.includes('增长率')) {{
                                result += '%';
                            }} else if (indicator.name.includes('波动性')) {{
                                result += ' kt CO₂当量';
                            }} else {{
                                result += ' kt CO₂当量';
                            }}
                            result += '<br/>';
                        }});
                        return result;
                    }}
                }},
                legend: {{
                    data: data.series.map(s => s.name),
                    top: 30,
                    type: 'scroll'
                }},
                radar: {{
                    indicator: data.indicators.map(ind => ({{
                        ...ind,
                        name: ind.name,
                        max: ind.max
                    }})),
                    radius: '60%',
                    center: ['50%', '55%']
                }},
                series: [{{
                    name: '排放指标',
                    type: 'radar',
                    data: data.series.map(s => ({{
                        name: s.name,
                        value: s.value,
                        areaStyle: {{ opacity: 0.3 }},
                        lineStyle: {{ width: 2 }}
                    }}))
                }}]
            }};

            chart.setOption(option);
        }})();'''

        return container, script

    def create_sankey_chart(self):
        print("    创建桑基图...")

        data = self.load_data_file('ECharts_桑基图数据.json')
        if not data:
            return None

        container = '''
            <div class="chart-container">
                <div class="chart-title">🌊 桑基图：排放量流向分析</div>
                <p class="chart-description">
                    <strong>ECharts优势：</strong>丰富的动画效果，出色的交互功能（悬停、点击、缩放），支持复杂数据流向展示
                </p>
                <div id="chartSankey" class="chart large"></div>
            </div>'''

        script = f'''
        // 桑基图
        (function() {{
            const data = {json.dumps(data, ensure_ascii=False)};
            const chart = echarts.init(document.getElementById('chartSankey'));

            const option = {{
                title: {{
                    text: data.title || '排放量流向分析',
                    left: 'center'
                }},
                toolbox: {{
                    feature: {{
                        saveAsImage: {{
                            name: '排放量流向桑基图',
                            type: 'png',
                            backgroundColor: '#fff'
                        }}
                    }}
                }},
                tooltip: {{
                    trigger: 'item',
                    triggerOn: 'mousemove',
                    formatter: function(params) {{
                        if (params.dataType === 'edge') {{
                            return params.data.source + ' → ' + params.data.target + '<br/>' +
                                   '排放量: ' + formatNumber(params.data.value) + ' kt CO₂当量';
                        }} else {{
                            return params.data.name;
                        }}
                    }}
                }},
                series: [{{
                    type: 'sankey',
                    data: data.nodes,
                    links: data.links,
                    nodeWidth: 20,
                    nodeGap: 8,
                    layoutIterations: 32,
                    orient: 'horizontal',
                    draggable: false,
                    focusNodeAdjacency: 'allEdges',
                    itemStyle: {{
                        borderWidth: 1,
                        borderColor: '#aaa'
                    }},
                    lineStyle: {{
                        color: 'gradient',
                        curveness: 0.5
                    }},
                    label: {{
                        fontSize: 12,
                        fontWeight: 'bold'
                    }},
                    emphasis: {{
                        focus: 'adjacency'
                    }}
                }}]
            }};

            chart.setOption(option);
        }})();'''

        return container, script

    def create_3d_surface_chart(self):
        print("    创建3D表面图...")

        # 加载3D表面图数据
        data = self.load_data_file('ECharts_3D表面图数据.json')
        if not data:
            return None

        container = '''
            <div class="chart-container">
                <div class="chart-title">🏔️ 3D表面图：主要国家排放量立体展示</div>
                <p class="chart-description">
                    <strong>ECharts优势：</strong>极强的3D交互性（旋转、缩放、平移），流畅渲染效果，实时调整视角和光照
                </p>
                <div id="surface3d" class="chart large"></div>
            </div>'''

        script = f'''
        // 3D表面图配置
        var surface3dChart = echarts.init(document.getElementById('surface3d'));

        var surface3dData = {json.dumps(data, ensure_ascii=False)};

        var surface3dOption = {{
            title: {{
                text: '{data.get("title", "3D表面图")}',
                left: 'center',
                textStyle: {{
                    fontSize: 18,
                    fontWeight: 'bold'
                }}
            }},
            toolbox: {{
                feature: {{
                    saveAsImage: {{
                        name: '3D表面图_排放量立体展示',
                        type: 'png',
                        backgroundColor: '#fff'
                    }}
                }}
            }},
            tooltip: {{
                trigger: 'item',
                formatter: function(params) {{
                    if (params.data) {{
                        var countryIndex = params.data[0];
                        var yearIndex = params.data[1];
                        var emission = params.data[2];
                        var country = surface3dData.xAxis.data[countryIndex];
                        var year = surface3dData.yAxis.data[yearIndex];
                        return country + '<br/>' + 
                               year + '年<br/>' + 
                               '排放量: ' + emission.toLocaleString() + ' kt CO₂当量';
                    }}
                    return '';
                }}
            }},
            grid3D: {{
                viewControl: {{
                    projection: 'perspective',
                    autoRotate: true,
                    autoRotateDirection: 'cw',
                    autoRotateSpeed: 5,
                    damping: 0.8,
                    rotateSensitivity: 1,
                    zoomSensitivity: 1,
                    panSensitivity: 1,
                    alpha: 30,
                    beta: 40,
                    center: [0, 0, 0],
                    distance: 200
                }},
                boxWidth: 200,
                boxHeight: 100,
                boxDepth: 80,
                light: {{
                    main: {{
                        intensity: 1.2,
                        shadow: true,
                        shadowQuality: 'high',
                        alpha: 40,
                        beta: 40
                    }},
                    ambient: {{
                        intensity: 0.3
                    }}
                }}
            }},
            xAxis3D: {{
                type: 'category',
                name: '{data.get("xAxis", {}).get("name", "国家")}',
                data: surface3dData.xAxis.data,
                axisLabel: {{
                    interval: 0,
                    rotate: 45,
                    textStyle: {{
                        fontSize: 10
                    }}
                }}
            }},
            yAxis3D: {{
                type: 'category', 
                name: '{data.get("yAxis", {}).get("name", "年份")}',
                data: surface3dData.yAxis.data
            }},
            zAxis3D: {{
                type: 'value',
                name: '{data.get("zAxis", {}).get("name", "排放量")}'
            }},
            series: [{{
                type: 'surface',
                data: surface3dData.data,
                shading: 'realistic',
                realisticMaterial: {{
                    roughness: 0.2,
                    metalness: 0.5
                }},
                postEffect: {{
                    enable: true,
                    SSAO: {{
                        enable: true,
                        intensity: 1.2,
                        radius: 5
                    }}
                }},
                itemStyle: {{
                    opacity: 0.8
                }},
                emphasis: {{
                    itemStyle: {{
                        opacity: 1
                    }}
                }}
            }}]
        }};

        surface3dChart.setOption(surface3dOption);

        // 响应式
        window.addEventListener('resize', function() {{
            surface3dChart.resize();
        }});
        '''

        return container, script

    def generate_complete_visualization(self):
        """生成优化的可视化HTML页面 - 只包含4个最适合ECharts的图表"""
        print("\n 生成ECharts优化可视化页面...")

        charts = [
            self.create_sankey_chart(),  # 桑基图 - ECharts优势：丰富动画和交互
            self.create_radar_chart(),  # 雷达图 - ECharts优势：强交互性和视觉效果
            self.create_timeseries_chart(),  # 动画时间序列 - ECharts优势：流畅动画
            self.create_3d_surface_chart()  # 3D表面图 - ECharts优势：强3D交互性
        ]

        # 过滤掉None值
        valid_charts = [chart for chart in charts if chart is not None]

        if not valid_charts:
            print(" 没有可用的图表数据")
            return

        # 组合容器和脚本
        containers = '\n'.join([chart[0] for chart in valid_charts])
        scripts = '\n\n'.join([chart[1] for chart in valid_charts])

        # 生成完整HTML
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = self.create_html_template().format(
            generation_time=generation_time,
            chart_containers=containers,
            chart_scripts=scripts
        )

        # 保存文件
        with open('ECharts_优化可视化分析.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(" 优化可视化页面已生成: ECharts_优化可视化分析.html")
        print(f"   包含 {len(valid_charts)} 个交互式图表")

        # 生成使用说明
        self.generate_usage_instructions()

    def generate_individual_charts(self):
        print("\n 生成优化图表的单独文件...")

        chart_methods = [
            ('sankey', '桑基图', self.create_sankey_chart),
            ('radar', '雷达图', self.create_radar_chart),
            ('timeseries', '动画时间序列', self.create_timeseries_chart),
            ('3d_surface', '3D表面图', self.create_3d_surface_chart)
        ]

        generated_files = []

        for chart_type, chart_name, method in chart_methods:
            chart_data = method()
            if chart_data:
                container, script = chart_data

                # 创建单独的HTML文件
                generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                html_content = self.create_html_template().format(
                    generation_time=generation_time,
                    chart_containers=container,
                    chart_scripts=script
                )

                filename = f'ECharts_优化_{chart_type}_{chart_name}.html'
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                generated_files.append(filename)
                print(f"   ✓ {chart_name}: {filename}")

        print(f" 共生成 {len(generated_files)} 个优化图表文件（专注ECharts优势）")


def main():
    """主函数"""
    print(" ECharts 优化可视化分析")
    print("专注于4个最适合ECharts的图表类型")
    print("=" * 60)
    print("选择的图表类型：")
    print("1.  桑基图 - 丰富动画效果和交互功能")
    print("2.  雷达图 - 强交互性和视觉效果")
    print("3.  动画时间序列 - 流畅动画和交互控制")
    print("4.  3D表面图 - 强3D交互性和渲染效果")
    print("=" * 60)

    visualizer = EChartsOptimizedVisualizer()

    try:
        # 生成完整可视化页面
        visualizer.generate_complete_visualization()

        print(f"\n ECharts优化可视化分析完成！")
        print(" 生成的文件:")
        print("    ECharts_优化可视化分析.html - 专业交互式分析页面")

    except Exception as e:
        print(f" 可视化过程出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()