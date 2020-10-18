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
    )

    fig.update_yaxes(
        zeroline=True
        , showgrid=True
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
        go.Scatter(
            y=Graph1[Target]*100
            , x=[Col.title() if type(Col) == "str" else Col for Col in Graph1.index.values]
            , name=Target
            , mode="lines"
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


class VariableBinning():
    '''
    Class to bin a variable according to a selection of metrics.
    :Attribute BinPlot: Plots Distribution vs. Event rate for a DataFrame with class Count columns and Event rate column.
    :Attribute BinVariable: Fits OptBinning algorithm and prints summary plot along with BinPlot (For visualising)
    :Attribute Transform: Fits and transforms variable, returns transformed series.

    '''
    
    def __init__(self, Df, Variable, Target, DType = "numerical"):
        self.Temp = Df.copy()[[Variable, Target, "Track"]]
        self.Variable = Variable
        self.Target = Target
        self.Mod = None
        self.DType = DType
    
    def BinPlot(self, Graph):
        '''
        :Param Graph: Dataframe containing Class count columns and event rate column
        '''
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
 
        fig.add_trace(
            go.Scatter(
                y=Graph["EventRate"]*100
                , x=[Col.title() if type(Col) == "str" else Col for Col in Graph.index.values]
                , name=self.Target
                , mode="lines+markers"
                , showlegend= True)
            , secondary_y = True)
    
        for Col in [str(self.Target)+" == 0", str(self.Target)+" == 1"]:
            fig.add_trace(
                go.Bar(
                    y=Graph[Col]
                    , x=[Col.title() if type(Col) == "str" else Col for Col in Graph.index.values]
                    , name=Col
                    , showlegend= True)
                , secondary_y = False)
 
        fig.update_xaxes(
            zeroline = True
            , showgrid = True
            , title = self.Variable
            , type='category' if self.DType == "categorical" else "-")
 
        fig.update_yaxes(
            zeroline=True
            , showgrid=True
            , title="Count"
            , secondary_y = False)
 
        fig.update_yaxes(
            zeroline=True
            , showgrid=False
            , title=self.Target
            , ticksuffix="%"
            , range=[0, 100]
            , secondary_y = True)
 
        fig.update_layout(
            title = dict(text= str(self.Variable) +" Distribution vs. " + str(self.Target), font=dict(color="Black", size=20))
            , font = dict(color="Black", size=10)
            , height = 600
            , width = 900
            , barmode='stack')
        
        fig.show()
        
        
    def BinVariable(self, Trend = "auto_asc_desc", Method = "bins", ShowTable = False):
        '''
        :Param Trend: Default = "auto_asc_desc", sets the assumed trend for binning
        :Param Method: Default = "bins", sets the desired transformation method.
        '''

        self.Mod = OptimalBinning(name=self.Variable, dtype=self.DType, solver="cp", max_n_prebins=100, monotonic_trend=Trend,
                                min_prebin_size=0.01, time_limit=30)
            
        #Fit and record
        self.Mod.fit(self.Temp[self.Variable], self.Temp[self.Target])
        
        BinningTable = self.Mod.binning_table
        Table = BinningTable.build()
        BinsValues = self.Mod.splits

        if ShowTable == True:
            #Print Status and Summary 
            print(self.Mod.status)
            print(BinningTable.analysis())
        
        self.Temp["Transformed"] = self.Mod.transform(self.Temp[self.Variable], metric = Method)

        if ((self.DType == "numerical") & (Method != "woe")):
            self.Temp["Transformed"] = self.Temp["Transformed"].apply(lambda s: tuple(float(x) for x in s.replace('[', '').replace(')', '').split(',')))
            self.Temp["Transformed"] = self.Temp["Transformed"].apply(lambda x: x[0] + 1 if math.isinf(x[1]) else x[1])
            
        Graph = pd.pivot_table(self.Temp, index="Transformed", columns=self.Target, values = "Track", aggfunc="count")
        Graph = Graph.rename({0: str(self.Target)+" == 0", 1: str(self.Target)+" == 1"}, axis=1)
        
        Graph1 = pd.pivot_table(self.Temp, index="Transformed", values=self.Target, aggfunc="mean")

        Graph["EventRate"] = Graph1[self.Target]
        
        self.BinPlot(Graph)
        
        
    def Transform(self, Df = None, Method = 'woe'):
        '''
        :Param Trend: Default = "auto_asc_desc", sets the assumed trend for binning
        :Param Method: Default = "bins", sets the desired transformation method.
        '''
        
        if Df is not None:
            DataFrame = Df.copy()
            DataFrame = DataFrame[[self.Variable]]
            
            DataFrame["Transformed"] = self.Mod.transform(DataFrame[self.Variable], metric = Method)
            
            
            return DataFrame["Transformed"]
        
        
        else:
            self.Temp["Transformed"] = self.Mod.transform(self.Temp[self.Variable], metric = Method)
            
            Graph = pd.pivot_table(self.Temp, index="Transformed", columns=self.Target, values = "Track", aggfunc="count")
            Graph = Graph.rename({0: str(self.Target)+" == 0", 1: str(self.Target)+" == 1"}, axis=1)

            Graph1 = pd.pivot_table(self.Temp, index="Transformed", values=self.Target, aggfunc="mean").sort_values(by=self.Target, ascending=False)

            Graph = Graph.reindex(Graph1.index)
            Graph["EventRate"] = Graph1[self.Target]

            self.BinPlot(Graph)
            
            return self.Temp["Transformed"]
            
        
        
        
  

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
    
    
    
def InformationValue(Df, Variable, Target):
    '''
    Computes the information for a given dataframe feature w.r.t a target/dependent variable.
    '''
    Pivot = pd.pivot_table(Df, index=Variable, values="Track", columns=Target, aggfunc="count").reset_index()
    Pivot = Pivot.rename({0:"Flops", 1:"Hits"}, axis=1)
    Pivot["Flops"] = Pivot["Flops"] / Pivot["Flops"].sum()
    Pivot["Hits"] = Pivot["Hits"] / Pivot["Hits"].sum() 

    Pivot["IV"] = Pivot["Flops"] - Pivot["Hits"] 
    Pivot["IV"] = Pivot["IV"]*Pivot[Variable]

    
    return Pivot["IV"].sum()