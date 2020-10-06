import pandas as pd
import numpy as np
import plotly.graph_objects as go
from optbinning import OptimalBinning
import math
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px


def DistributionPlot(Df, PlotVar):
    '''
    Plots the distribution of a given variable in a dataframe
    '''
    Labels = [i for i in range(0, 100, 10)]
    
    BinSize = Df[PlotVar].describe().loc["std"] / 20
    
    fig = ff.create_distplot(
            hist_data = [Df[Df["Target"] == 0][PlotVar].values, Df[Df["Target"] == 1][PlotVar].values]
            , group_labels = [0, 1]
            , bin_size=BinSize
            , show_hist=True)
    
    fig.update_xaxes(
    zeroline = True
    , showgrid = True
    , title=PlotVar)


    fig.update_yaxes(
        zeroline=True
        , showgrid=True
        , title="Distribution")


    fig.update_layout(
        title = dict(text=str(PlotVar) + " Distribution"
                     , font=dict(color="Black", size=20))
        , font = dict(color="Black", size=10)
        , height = 700
        , width = 1100
        , legend_title='Target')

    fig.show(renderer='png', height=700, width=1100)




def Scatter(Df, PlotVar, Hue, Y, Title):
    
    '''
    Produces a plot of data pulled from specified dataframe split by a certain binary population
    PlotVars defines the independent variable
    Hue defines the population for which to split the plots
    Y is the dependent variable
    Title is title of the plot
    '''

    fig = go.Figure()

    fig.add_trace(  
        go.Scatter(
            x = Df[Df[Hue] == 1][PlotVar]
            , y=Df[Df[Hue] == 1][Y]
            , legendgroup=Hue + " = 1"
            , name=Hue + " = 1"
            , mode='markers'
            , line=dict(color='red')
            , marker=dict(size=10, opacity=0.1)
            , showlegend= True))

    fig.add_trace(  
        go.Scatter(
            x = Df[Df[Hue] == 0][PlotVar]
            , y=Df[Df[Hue] == 0][Y]
            , legendgroup=Hue + " = 0"
            , name=Hue + " = 0"
            , mode='markers'
            , line=dict(color='blue')
            , marker=dict(size=10, opacity=0.1)
            , showlegend= True))

    fig.update_xaxes(
        zeroline = True
        , showgrid = True
        , title = PlotVar
        #, range = [0.95*np.min(Df[PlotVar]), 1.05*np.max(Df[PlotVar])]
    )

    fig.update_yaxes(
        zeroline=True
        , showgrid=True
        #, range = [0.95*np.min(Df[Y]), 1.05*np.max(Df[Y])]
        , title = Y)
    
    
    fig.update_layout(
        title = dict(text=Title, font=dict(size=17)))

    fig.update_annotations(
        font = dict(size=14))
    
    fig.show(renderer="png", height=600, width=1000)



def Distribution(Df, Target, Variable):    
    Graph = pd.pivot_table(Df, index=Variable, columns=Target, values="Track", aggfunc=len)
    Graph1 = pd.pivot_table(Df, index=Variable, values=Target, aggfunc="mean").sort_values(by="Target", ascending=False)
    
    Graph = Graph.reindex(Graph1.index)


    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(
        go.Line(
            y=Graph1[Target]*100
            , x=[Col.title() if type(Col) == "str" else Col for Col in Graph1.index.values]
            , name=Target
            , showlegend= True)
        , secondary_y = True)


    fig.add_trace(
        go.Bar(
            y=Graph[0]
            , x=[Col.title() if type(Col) == "str" else Col for Col in Graph1.index.values]
            , name="Not"+str(Target)
            , showlegend= True)
        , secondary_y = False)

    fig.add_trace(
        go.Bar(
            y=Graph[1]
            , x=[Col.title() if type(Col) == "str" else Col for Col in Graph1.index.values]
            , name=Target
            , showlegend= True)
        , secondary_y = False)


    fig.update_xaxes(
        zeroline = True
        , showgrid = True
        , title = Variable
        , type="category")

    fig.update_yaxes(
        zeroline=True
        , showgrid=True
        , title="Frequency"
        , secondary_y = False)

    fig.update_yaxes(
        zeroline=True
        , showgrid=False
        , title=Target
        , ticksuffix="%"
        , range=[0, 100]
        , secondary_y = True)


    fig.update_layout(
        title = dict(text= str(Variable) +" Distribution vs. " + str(Target), font=dict(color="Black", size=20))
        , font = dict(color="Black", size=10)
        , height = 600
        , width = 900
        , barmode='stack')


    fig.update_annotations(
    font = dict(color="Black", size=14))

    fig.show(renderer="png", width=900, height=600)



def BinVariable(Df, Target, Variable, BinOut, Trend = "auto_asc_desc"):
    Size = 0.01 #Picks minimum bin size
    
    OptB = OptimalBinning(name=Variable, dtype="numerical", solver="cp", max_n_prebins=100, monotonic_trend=Trend,
                          min_prebin_size=Size, time_limit=30)

    OptB.fit(Df[Variable], Df[Target])
    
    print(OptB.status)
    
    BinsValues = OptB.splits #Where the splits are
    BinningTable = OptB.binning_table
    Bins = BinningTable.build()
    Analysis = BinningTable.analysis() #Summary
    
    print(Analysis)

    BinOut[Variable] = BinsValues

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            y=Bins["Non-event"][0:len(BinsValues)+1]
            , x=Bins["Bin"][0:len(BinsValues)+1]
            , name="Frequency " + str(Target) + " = 0"
            , showlegend= True)
        , secondary_y = False)

    fig.add_trace(
        go.Bar(
            y=Bins["Event"][0:len(BinsValues)+1]
            , x=Bins["Bin"][0:len(BinsValues)+1]
            , name="Frequency " + str(Target) + " = 1"
            , showlegend= True)
        , secondary_y = False)

    fig.add_trace(
        go.Line(
            y=Bins["Event rate"][0:len(BinsValues)+1]*100
            , x=Bins["Bin"][0:len(BinsValues)+1]
            , name=str(Target) + " Rate"
            , showlegend= True
            , connectgaps = True)
        , secondary_y = True)


    fig.update_xaxes(
        zeroline = True
        , showgrid = True
        , title = Variable
        , tickmode = 'linear')


    fig.update_yaxes(
        zeroline=True
        , showgrid=True
        , title="Frequency"
        , secondary_y = False)

    fig.update_yaxes(
        zeroline=True
        , showgrid=False
        , title=str(Target) + " Rate"
        , ticksuffix="%"
        , range=[0, 100]
        , secondary_y = True)


    fig.update_layout(
        title = dict(text= Variable +" Distribution vs. " + str(Target), font=dict(color="Black", size=20))
        , font = dict(color="Black", size=10)
        , height = 600
        , width = 900
        , legend_title='Period'
        , barmode='stack')


    fig.update_annotations(
    font = dict(color="Black", size=14))

    fig.show(renderer="png", width=900, height=600)
    
    return BinOut


def Correlation(Df, PlotVars, Title): 
    
    '''
    Provides a correlation matrix heatmap for data pulled from a specified dataframe
    PlotVars define all features in question
    Title is title of the plot
    '''

    Correlations = Df[PlotVars].corr()

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=Correlations
            , x=Correlations.index
            , y=Correlations.index
            , zmax=1
            , zmin=-1
            , hoverongaps = False
            , colorscale=[(0, "blue"), (0.5, "white"), (1, "red")]))


    fig.update_layout(
        title = dict(text=Title, font=dict(color="Black", size=20))
        , font = dict(color="Black", size=10)
        , height = 1000
        , width = 1000
        , legend_title='Period')


    fig.update_annotations(
    font = dict(color="Black", size=14))

    fig.show(renderer="png", height=900, width=900)


def BarPlot(DataFrame, Title):
    
    fig = go.Figure()
    
    for Column in DataFrame.columns.values:
        
        fig.add_trace(
            go.Bar(
                y=DataFrame[Column]
                , x=DataFrame.columns.values
                , name=str(Column)
                , showlegend= True))


    fig.update_xaxes(
        zeroline = True
        , showgrid = True
        , title = "Features"
        , showticklabels=False)

    fig.update_yaxes(
        zeroline=True
        , showgrid=True
        , title="Importance")


    fig.update_layout(
        title = dict(text= Title, font=dict(color="Black", size=20))
        , font = dict(color="Black", size=10)
        , height = 600
        , width = 900)


    fig.update_annotations(
    font = dict(color="Black", size=14))

    fig.show(renderer="png", width=900, height=600)
