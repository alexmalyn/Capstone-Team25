import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pickle
import numpy as np
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#1a1a1a',
    'text': '#f2f2f2',
    'textShadow': '2px 2px 2px #0d0d0d'
}

data = pd.read_pickle('./storage/1yeardata/data.pickle')
ex_data = data['AAPL']
#fig = px.line(data)

realclose = np.load('./storage/vis/realclose.pickle',allow_pickle=True)

predictions_1 = np.load('./storage/vis/predictionsclose_1_epochs.pickle',allow_pickle=True)
predictions_5 = np.load('./storage/vis/predictionsclose_5_epochs.pickle',allow_pickle=True)
predictions_10 = np.load('./storage/vis/predictionsclose_10_epochs.pickle',allow_pickle=True)

discloss_1 = np.load('./storage/vis/discloss_1_epochs.pickle',allow_pickle=True)
genloss_1 = np.load('./storage/vis/genloss_1_epochs.pickle',allow_pickle=True)
discloss_5 = np.load('./storage/vis/discloss_5_epochs.pickle',allow_pickle=True)
genloss_5 = np.load('./storage/vis/genloss_5_epochs.pickle',allow_pickle=True)
discloss_10 = np.load('./storage/vis/discloss_10_epochs.pickle',allow_pickle=True)
genloss_10 = np.load('./storage/vis/genloss_10_epochs.pickle',allow_pickle=True)

fig1 = px.line(genloss_1,labels={'x-label': 'Training Iterations'})

fig1.update_layout(
    title='Generator Loss',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=False,  
)

fig1.update_xaxes(title={'text':'Training Iterations'})

fig2 = px.line(discloss_1)

fig2.update_layout(
    title='Discriminator Loss',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=False,
)

fig2.update_xaxes(title={'text':'Training Iterations'})

fig3 = px.line(genloss_5,labels={'x-label': 'Training Iterations'})

fig3.update_layout(
    title='Generator Loss',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=False,
)

fig3.update_xaxes(title={'text':'Training Iterations'})

fig4 = px.line(discloss_5)

fig4.update_layout(
    title='Discriminator Loss',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=False,
)

fig4.update_xaxes(title={'text':'Training Iterations'})

fig5 = px.line(genloss_10,labels={'x-label': 'Training Iterations'})

fig5.update_layout(
    title='Generator Loss',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=False,
)

fig5.update_xaxes(title={'text':'Training Iterations'})

fig6 = px.line(discloss_10)

fig6.update_layout(
    title='Discriminator Loss',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=False,
)

fig6.update_xaxes(title={'text':'Training Iterations'})

fig7data = pd.DataFrame({'Real': realclose, 'Forecasted': predictions_1})

fig7 = px.line(fig7data)

fig7.update_layout(
    title='Training',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=True,
)

fig7.update_xaxes(title={'text':'Trading Hours'})

fig8data = pd.DataFrame({'Real': realclose, 'Forecasted': predictions_5})

fig8 = px.line(fig8data)

fig8.update_layout(
    title='Training',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=True,
)

fig8.update_xaxes(title={'text':'Trading Hours'})

fig9data = pd.DataFrame({'Real': realclose, 'Forecasted': predictions_10})

fig9 = px.line(fig9data)

fig9.update_layout(
    title='Training',
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],
    showlegend=True,
)

fig9.update_xaxes(title={'text':'Trading Hours'})

def ex_data_func(dataframe,rows):

    dataframe = dataframe.reset_index()
    dataframe = dataframe.round(decimals=3)
        
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(5)
        ])
    ],
        style={
            
            'color': colors['text'],
            'margin-top':'0px',
            'margin-left': '50px',
            'overflowX':'scroll',
            #'display':'inline-block'
        })

app.layout = html.Div([

    html.H1(   
        'Deep Learning Market Analysis',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin-top':'0px',
            'padding-top': '20px',
            'textShadow': colors['textShadow']
        }  
    ),
    
    html.H3(
        'Visualizations',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '0px',
            #'display':'inline-block',
            'textShadow': colors['textShadow']
        } 
    ),
    
    
    #DATA
    html.H3(
        '> Data Used',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block',
            'textShadow': colors['textShadow']}),
    html.P('S&P 500 ticker data over the last year',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '50px',
            #'display':'inline-block',
            'textShadow': colors['textShadow']}),
    html.P('Ex: AAPL Stock Data,',
    style={
        'textAlign': 'left',
        'color': colors['text'],
        'margin-top':'20px',
        'margin-left': '50px',
        #'display':'inline-block',
        'textShadow': colors['textShadow']}),

    ex_data_func(ex_data,5),
    
    #INDICATORS DERIVED
    html.H3(
        '> Technical Indicators',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block',
            'textShadow': colors['textShadow']
        } 
    ),
    
    html.Img(src=app.get_asset_url('ti-horiz.png'),
        style={
            'margin-left': '35px',
        }
    ),
    
    #LSTM-GAN Architecture
    html.H3(
        '> GAN Architecture',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block',
            'textShadow': colors['textShadow']
        } 
    ),
    
    html.Img(src=app.get_asset_url('gen_archi.png'),
        style={
            'margin-left': '50px',
            'height':'500px',
            'width': '300px'
        }),
    html.Img(src=app.get_asset_url('disc_archit.png'),
        style={
            'margin-left': '50px',
            'height':'500px',
            'width': '300px'
        }),
    html.Img(src=app.get_asset_url('gan_archi.png'),
        style={
            'margin-left': '50px',
            'height':'400px',
            'width': '500px'
        }),
    
    #Step-Epoch fitting
    html.H3(
        '> Training',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block',
            'textShadow': colors['textShadow']} 
    ),
    html.P('Example: AAPL',
    style={
        'textAlign': 'left',
        'color': colors['text'],
        'margin-top':'20px',
        'margin-left': '50px',
        #'display':'inline-block',
        'textShadow': colors['textShadow']}),
    
    dcc.Dropdown(
        id='epoch-dropdown',
        options=[
            {'label': '1 epoch', 'value': 1},
            {'label': '5 epochs', 'value': 5},
            {'label': '10 epochs', 'value': 10},
        ], value=1, searchable=False, clearable=False,
        style = {
            'margin-left' : '25px',
            'width': '150px',
            #'backgroundColor' : colors['background'],
            #'color' : colors['text'],
        }
    ),
    
    html.Div([
        dcc.Graph(id = 'gen-graph',
            figure = fig1, className='six columns'
        ),
        dcc.Graph(id = 'disc-graph',
            figure = fig2, className='six columns'
        ),
    ], className='row', style = {
        'margin-left': '30px'}),
        
    html.Div([
        dcc.Graph(id = 'training-graph',
            figure = fig7
        ),
    ], className='row', style = {
        'margin-left': '30px'}),
    
    #TRENDS
    html.H3(
        '> GICS Trends',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block',
            'textShadow': colors['textShadow']
        } 
    ),
    
    ],
    style={
        'backgroundColor': colors['background'],
        #'verticalAlign':'middle',
        #'textAlign': 'center',
        #'position':'fixed',
        #'width':'100%',
        #'height':'100%',
        #'top':'0px',
        #'left':'0px',
        #'z-index':'1000'
    }
)

@app.callback(
    Output('gen-graph', 'figure'),
    Output('disc-graph','figure'),
    Output('training-graph','figure'),
    Input('epoch-dropdown', 'value'))
def update_graphs(value):
    if value == 1:
        return fig1, fig2, fig7
    if value == 5:
        return fig3, fig4, fig8
    if value == 10:
        return fig5, fig6, fig9


if __name__ == '__main__':
    app.run_server(debug=True)