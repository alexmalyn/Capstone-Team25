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
    'background': '#111111',
    'text': '#FFFFFF'
}

data = pd.read_pickle('./storage/1yeardata/data.pickle')
ex_data = data['AAPL']
#fig = px.line(data)

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
            'margin-top':'20px'
        }  
    ),
    
    html.H3(
        'Visualization Tool',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '0px',
            #'display':'inline-block'
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
            #'display':'inline-block'
        } 
    ),
    html.P('S&P 500 ticker data over the last year',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '50px',
            #'display':'inline-block'
        }),
    html.P('AAPL Stock Data:',
    style={
        'textAlign': 'left',
        'color': colors['text'],
        'margin-top':'20px',
        'margin-left': '50px',
        #'display':'inline-block'
    }),

    ex_data_func(ex_data,5),
    
    #INDICATORS DERIVED
    html.H3(
        '> Technical Indicators',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block'
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
            #'display':'inline-block'
        } 
    ),
    
    #Step-Epoch fitting
    html.H3(
        '> Training',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block'
        } 
    ),
    
    #TREND FINALE
    html.H3(
        '> GICS Trends',
        style={
            'textAlign': 'left',
            'color': colors['text'],
            'margin-top':'20px',
            'margin-left': '25px',
            #'display':'inline-block'
        } 
    ),
    
    ],
    style={
        'backgroundColor': colors['background'],
        'verticalAlign':'middle',
        'textAlign': 'center',
        'position':'fixed',
        'width':'100%',
        'height':'100%',
        'top':'0px',
        'left':'0px',
        'z-index':'1000'
    }
)




if __name__ == '__main__':
    app.run_server(debug=True)