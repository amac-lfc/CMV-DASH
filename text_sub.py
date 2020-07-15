from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4"))

fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
              row=1, col=1)

fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
              row=1, col=2)

fig.add_trace(go.Scatter(x=[300, 400, 500], y=[600, 700, 800]),
              row=2, col=1)

fig.add_trace(go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000]),
              row=2, col=2)

fig.update_layout(height=500, width=700,
                  title_text="Multiple Subplots with Titles")


annotations = []
annotations.append(
        dict(
                                x=2,
                                y=5,
                                xref="x1",
                                yref="y1",
                                text="Your comment",
                                showarrow=True,
                                arrowhead=7,
                                ax=0,
                                ay=-40,
                                arrowcolor='red',
                                font=dict(
                                family="Courier New, monospace",
                                size=16,
                                color="#000"
                                ),
                                align="center",
                                bordercolor="red",
                                borderwidth=2,
                                borderpad=4,
                                bgcolor="red",
                                opacity=0.8
        ))

fig['layout'].update(annotations=annotations)

fig.show()