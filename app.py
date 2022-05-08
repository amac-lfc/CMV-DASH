import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import features
import plotly.tools as tls
import lib

external_stylesheets = ['https://fonts.googleapis.com/css?family=Lato',\
        'https://fonts.googleapis.com/css?family=Permanent+Marker&display=swap']

feat_list = np.array(['lexical diversity', 'character count', '# of links', '# of quotes', '# of questions', \
                     '# of bold characters', 'Average Sentence length', '# of enumerations', '# of exclamation point', '# of certainty word', \
                     '# of extremity words', '# of high arousal words', '# of low arousal words', '# of medium arousal words', '# of medium dominance words', \
                     '# of low dominance words', '# of high dominance words', '# of high valence words', '# of low valence words', '# of medium valence words', '# of examples words', '# of hedges words',\
                      '# of self references words'])

means = np.array([[7.28019324e+01, 1.36948275e+01],
 [1.13771958e+03, 1.30121935e+03],
 [2.80346599e-01, 1.17337788e+00],
 [6.23648493e-01, 1.61981222e+00],
 [1.01242236e+00, 1.99393274e+00],
 [2.66313933e-01, 4.85656618e+00],
 [1.59775324e+01, 7.88181633e+00],
 [5.88145081e-02, 3.23395602e-01],
 [1.28211027e-01, 5.38355635e-01],
 [1.65455103e+00, 2.55759142e+00],
 [2.57679626e+00, 3.59872152e+00],
 [6.06648263e+00, 7.64842330e+00],
 [7.68069933e+00, 8.72348011e+00],
 [3.44532628e+01, 3.64271447e+01],
 [2.94032666e+01, 3.14786827e+01],
 [4.03780385e+00, 5.58305384e+00],
 [1.47600644e+01, 1.61387337e+01],
 [1.26744115e+01, 1.42055133e+01],
 [4.31500652e+00, 5.83298959e+00],
 [3.12124070e+01, 3.34550477e+01],
 [7.13135496e-03, 1.92716225e-01],
 [3.29322905e+00, 4.06426683e+00],
 [1.43386243e+00, 3.04885967e+00]])

example = '''From a statistical perspective, sure. But would you agree that a rollercoaster is a more fearful experience than driving? The odds of dying in a rollercoaster are exponentially less. However, the feeling of moving very fast, being high off the ground, and being unable to stop and leave when you want to, are all likely to trigger discomfort since, on most other occasions, moving fast, being up high, or being trapped mean we are more vulnerable. These same factors also apply to flying. In this sense, people's responses are rational in the sense that there is a reasonable past source behind certain feelings or bodily responses'''

model = load('GradientBoosting.joblib') 
word_list_input = "word_list.csv"

app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.title = 'CMV Check' 

server = app.server

app.layout = html.Div([
    html.Div(html.H1("CMV Check"), style={'text-align': 'center','margin': '0 auto'}),
 html.Div(dcc.Markdown('''
*Authors: Arthur Bousquet, Leo Carrico, Vivian Ta.* You can learn more on our [Github](https://github.com/amac-lfc/CMV). 
See more porjects at [AMAC](http://amac.xyz)'''),style={'text-align': 'center','margin': '0 auto'}) ,
    html.Div(html.P('Check if the following text is more likely to ended chaning the mind of the person\
        (obtainind a delta) for the subreddit Change My View:'), style={'margin-bottom': '2em', 'margin-top': '2em'}),
    dcc.Textarea(
        id='textarea',
        value=example,
        style={'width':'100%', 'height': 300, 'margin': '0 auto'},
    ),
    html.Div(html.Button('Submit', id='textarea-button', n_clicks=0),\
        style={'text-align': 'left','margin-bottom': '2em','margin-botom': '5em'}),
    html.Div(html.P('Probability that is a Delta:'), style={'text-align': 'center'}),
    html.Div(id='textarea-output', style={'text-align': 'center','margin': '0 auto','width':'11em',}),
    html.Div(html.P('You can compare your text to the average fo each features of other delta comments.')\
        , style={'margin-bottom': '2em','margin-top': '5em'}),
    dcc.Graph(id='graph1', style={'margin': '0 auto'}),
], style={'max-width': '60em', 'margin': '0 auto'})

@app.callback(
    Output('textarea-output', 'children'),
    [Input('textarea-button', 'n_clicks')],
    [State('textarea', 'value')]
)
def update_output(n_clicks, value):
    #  if n_clicks > 0:
    # return 'You have entered: \n{}'.format(value)
    # intialise data of lists. 
    data = {'body':[value]} 
    # Create DataFrame 
    df = pd.DataFrame(data) 
    X=features.generateLanFeatures(df, word_list_input, 'con')
    X=model.predict_proba(X)
    # return 'Probabilties it is not a delta {:.2f} and it is a delta {:.2f}'.format(X[0,0],X[0,1])
    if X[0,1]>0.5:
        return html.Div([html.Div([html.P('{:.2f}'.format(X[0,1]))],className="results"),"It is a Delta!"])
    else:
        return html.Div([html.Div([html.P('{:.2f}'.format(X[0,1]))],className="results-neg"),"It is NOT a Delta!"])


@app.callback(
    Output('graph1', 'figure'),
    [Input('textarea-button', 'n_clicks')],
    [State('textarea', 'value')]
)
def update_graph1(n_clicks, value):
    # if n_clicks > 0:
    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    print("lexi =",lib.getLexicalDiversity(value))
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=0
    # trace = []
    sub_titles = []
    for i in range(11):
        sub_titles.append(feat_list[i])
        sub_titles.append(feat_list[10+i+1])
    sub_titles.append(feat_list[22])
    # sub_titles=np.array(sub_titles)
    print(sub_titles)
    print(len(sub_titles))
    # sub_titles=('toto','tata')
    fig = tls.make_subplots(rows=12, cols=2,subplot_titles=sub_titles,horizontal_spacing=0.08,vertical_spacing=0.05)
    annotations = []

    for i in range(12):
        # print("row=",i+1,i)
        # print("x"+str(i+1))
        fig.append_trace(go.Scatter(y=[0,0,0],x=[means[i,0]-2*means[i,1],means[i,0],means[i,0]+2*means[i,1]],mode='markers+lines+text', \
                text=["m-2s", "m", "m+2s"], textposition ="bottom center", marker_size=10, marker_color='#5bc783', textfont = {'family': "Lato", 'size': [15,15,15], 'color': ["# 7ccc7c","# 7ccc7c","# 7ccc7c"]}),row=i+1, col=1)
        fig.append_trace(go.Scatter(y=[0],x=[feat[0,i]],mode='markers+text',text=['your<br>text'],textposition ="top center", marker_size=10,marker_color='#9d6ace',textfont = {'family': "Lato", 'size': [20], 'color': ["#9d6ace"]}),row=i+1, col=1)

        # annotations.append(
        # dict(
        #                         x=feat[0,i],
        #                         y=0,
        #                         xref="x"+str(2*i+1),
        #                         yref="y"+str(2*i+1),
        #                         text="Your comment",
        #                         showarrow=True,
        #                         arrowhead=7,
        #                         ax=0,
        #                         ay=-40,
        #                         arrowcolor='red',
        #                         font=dict(
        #                         family="Courier New, monospace",
        #                         size=16,
        #                         color="# 7ccc7c"
        #                         ),
        #                         align="center",
        #                         bordercolor="red",
        #                         borderwidth=2,
        #                         borderpad=4,
        #                         bgcolor="red",
        #                         opacity=0.8
        # ))


        

    for i in range(12,23):
        # print("row=",i-11,i)
        # print("x"+str(i-11))
        fig.append_trace(go.Scatter(y=[0,0,0],x=[means[i,0]-2*means[i,1],means[i,0],means[i,0]+2*means[i,1]],mode='markers+lines+text', \
                text=["m-2s", "m", "m+2s"], textposition ="bottom center", marker_size=10, marker_color='#5bc783', textfont = {'family': "Lato", 'size': [15,15,15], 'color': ["# 7ccc7c","# 7ccc7c","# 7ccc7c"]}),row=i-11, col=2)
        fig.append_trace(go.Scatter(y=[0],x=[feat[0,i]],mode='markers+text',text=['your<br>text'],textposition ="top center", marker_size=10,marker_color='#9d6ace', textfont = {'family': "Lato", 'size': [20], 'color': ["#9d6ace"]}),row=i-11, col=2)

        # annotations.append(
        # dict(
        #                         x=feat[0,i],
        #                         y=0,
        #                         xref="x"+str(2*(i-11)),
        #                         yref="y"+str(2*(i-11)),
        #                         text="Your comment",
        #                         showarrow=True,
        #                         arrowhead=7,
        #                         ax=0,
        #                         ay=-40,
        #                         arrowcolor='red',
        #                         font=dict(
        #                         family="Courier New, monospace",
        #                         size=16,
        #                         color="# 7ccc7c"
        #                         ),
        #                         align="center",
        #                         bordercolor="red",
        #                         borderwidth=2,
        #                         borderpad=4,
        #                         bgcolor="red",
        #                         opacity=0.8
        # ))
    # fig.append_trace(go.Scatter(y=[0],x=[0],mode='markers'),row=12, col=2)
    # i=2
    # fig.append_trace(go.Scatter(y=[0,0,0],x=[means[i,0]-2*means[i,1],means[i,0],means[i,0]+2*means[i,1]],mode='markers+lines+text', \
    #         text=["2 std from the mean", "mean", "2 std from the mean"], textposition ="bottom center", marker_size=10, hovertext=None),row=1, col=2)
    # fig.append_trace(go.Scatter(y=[0],x=[feat[0,i]],mode='markers+text',text=['your comment'],textposition ="top center", marker_size=10,marker_color='red'),row=1, col=2)

    # print("########## feat=",feat[0,i])

    # fig['layout'].update(annotations=annotations)
    fig.update_yaxes(showticklabels=False)

    # to change subtitle, address subplot
    # fig['layout']['annotations'][0].update(text='your text here');
    # fig['layout']['annotations'][1].update(text='your text here');

    fig.update_layout(showlegend=False)
    fig.update_layout(height=3000, width=1000, plot_bgcolor='white',title_text="Features")





    # trace.append(go.Scatter(y = list(delta_data.columns), x = feat[0],mode = 'markers',name = 'your commemt'))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)