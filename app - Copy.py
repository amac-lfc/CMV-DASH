import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import features

feat_list = np.array(['lexical_diversity', 'char_count_rounded', 'link_count', 'quote_count', 'questions_count', \
                     'bold_count', 'avgSentences_count', 'enumeration_count', 'excla_count', 'certainty_count', \
                     'extremity_count', 'high arousal', 'low arousal', 'medium arousal', 'medium dominance', \
                     'low dominance', 'high dominance', 'high valence', 'low valence', 'medium valence', 'examples', 'hedges', 'self references'])

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

example = '''I believe you are missing the point of gender. The notion of gender is a social construct. It is not something you can change. It is a construct that can and should be changed. In my opinion, gender is not a construct you can actually change. 
It is not something that people are forced to "change". It is a set of behaviors that people can and should mimic.
The issue with this is that you are saying that being forced to be a woman is like being forced to be a man. I am not talking about "being forced to be a man..." 
but "being forced to be a woman is like being forced to be a man." I would like to think you are aware of this, but I am not. I am a man, and I am talking about 
being forced to be a man in a non-threatening manner. That does not make it any less "manly." It just makes it a little more "manly." This is a universal experience, 
and one that many trans people are forced to face. I am not saying you should not be forced to be a woman, but I am saying you should be forced to be a man. I am not 
talking here about "being forced to be a man is wrong" or "being forced to be a man is wrong." I am not saying it is okay to be forced to be a man in public, but I am 
saying it is ok to be forced to be a woman in private. Just think about how uncomfortable it is for you to be forced to be a man in public, and how uncomfortable 
it will be for you to be forced to be a woman in the same manner. I am willing to bet you that many trans people will not be forced to be a woman, but many will. 
If you are truly bothered by this, you need to consider what the purpose of the feminine would be in relation to the masculine'''


model = load('GradientBoosting.joblib') 
word_list_input = "word_list.csv"

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div(html.P("Check if a CMV comment will give you a delta or not:"), style={'width': '50%', 'margin': '0 auto'}),
    dcc.Textarea(
        id='textarea-example',
        value=example,
        style={'width': '100%', 'height': 300},
    ),
    html.Div(id='textarea-example-output', style={'width': '50%', 'margin': '0 auto'}),
    dcc.Graph(id='graph1', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph2', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph3', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph4', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph5', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph6', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph7', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph8', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph9', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph10', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph11', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph12', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph13', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph14', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph15', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph16', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph17', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph18', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph19', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph20', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph21', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph22', style={'width': '100%','height': 300}),
    dcc.Graph(id='graph23', style={'width': '100%','height': 300}),
], style={'width': '50%', 'margin': '0 auto'})

@app.callback(
    Output('textarea-example-output', 'children'),
    [Input('textarea-example', 'value')]
)
def update_output(value):
     # intialise data of lists. 
    data = {'body':[value]} 
    # Create DataFrame 
    df = pd.DataFrame(data) 
    X=features.generateLanFeatures(df, word_list_input, 'con')
    X=model.predict_proba(X)
    return 'Probabilties it is not a delta {:.2f} and it is a delta {:.2f}'.format(X[0,0],X[0,1])


@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=0
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=1
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph3', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=2
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph4', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=3
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph5', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=4
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph6', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=5
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph7', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=6
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph8', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=7
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph9', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=8
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph10', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=9
    return update_feature(feat,i)


@app.callback(
    dash.dependencies.Output('graph11', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=10
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph12', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=11
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph13', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=12
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph14', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=13
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph15', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=14
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph16', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=15
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph17', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=16
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph18', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=17
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph19', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=18
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph20', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=19
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph21', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=20
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph22', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=21
    return update_feature(feat,i)

@app.callback(
    dash.dependencies.Output('graph23', 'figure'),
    [dash.dependencies.Input('textarea-example', 'value')])
def update_graph1(value):

    # print(list(delta_data.columns))
    data = {'body':[value]} 
    df = pd.DataFrame(data) 
    feat=features.generateLanFeatures(df, word_list_input, 'con')
    
    i=22
    return update_feature(feat,i)
    
def update_feature(feat,i):




    trace = []

    trace.append(go.Scatter(y=[0,0,0],x=[means[i,0]-2*means[i,1],means[i,0],means[i,0]+2*means[i,1]],mode='markers+lines', \
            text=["2 std from the mean", "mean", "2 std from the mean"], textposition ="top center", marker_size=10, hovertext=None))
    trace.append(go.Scatter(y=[0],x=[feat[0,i]],mode='markers',textposition ="top center", marker_size=10,marker_color='red'))

    print("########## feat=",feat[0,i])

    annotation = []
    annotation.append(dict(
                            x=means[i,0],
                            y=0,
                            xref="x",
                            yref="y",
                            text="mean",
                            showarrow=True,
                            arrowhead=7,
                            ax=0,
                            ay=-40
                        ))
    annotation.append(dict(
                            x=means[i,0]-2*means[i,1],
                            y=0,
                            xref="x",
                            yref="y",
                            text="mean-2*std",
                            showarrow=True,
                            arrowhead=7,
                            ax=0,
                            ay=-40
                        ))
    annotation.append(dict(
                            x=means[i,0]+2*means[i,1],
                            y=0,
                            xref="x",
                            yref="y",
                            text="mean+2*std",
                            showarrow=True,
                            arrowhead=7,
                            ax=0,
                            ay=-40
                        ))
    annotation.append(dict(
                            x=feat[0,i],
                            y=0,
                            xref="x",
                            yref="y",
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
    


    # trace.append(go.Scatter(y = list(delta_data.columns), x = feat[0],mode = 'markers',name = 'your commemt'))
    return{
        'data': trace,
        # 'annotation': annotation,
        'layout':
            go.Layout(title = feat_list[i],
                     showlegend=False,
                     annotations=annotation,
                     yaxis=dict(showticklabels=False)
                    #   xaxis=dict(range=[-6, 15])
                      )
        }


if __name__ == '__main__':
    app.run_server(debug=True)