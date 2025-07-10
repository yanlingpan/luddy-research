from dash import Dash, dcc, html, Input, Output, Patch

import os
import random
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict 
from sklearn.manifold import MDS


# bubble plot data
data_dir = Path("./data")
df = pd.read_csv(data_dir.joinpath("area2category_score_campus.csv"), index_col=["campus", "area_shortname", "area"])
df_norm = df.div(df.sum(axis=1), axis=0)
df['category'] = df.idxmax(axis=1)
bubble_size = 60
font_size = 8
df['size'] = bubble_size # bubble size
df = df.reset_index()
df['area_campus'] = df['area_shortname'] + "<br>(" + df['campus'] + ")"


# embed score w/ MDS
# mds_seed = random.randint(0, 10000)
# print(f"mds random seed: {mds_seed}")
mds_seed = 2971
embedding = MDS(n_components=2, n_init=4, random_state=mds_seed).fit_transform(df_norm)
embedding_df = pd.DataFrame(embedding, columns=["x", "y"])
embedding_df = (embedding_df-embedding_df.min())/(embedding_df.max()-embedding_df.min())
embedding_df = pd.concat([embedding_df, df[["area_campus", "campus", 'area', 'area_shortname', 'category', 'size']]], axis=1)

# extra info: area pis & links
area2pi2url = pd.read_csv(data_dir.joinpath("area2pi2url.csv"))
area2pis_dict = defaultdict(list)
for area in area2pi2url['area'].unique():
  area2pis_dict[area] = area2pi2url[area2pi2url['area'] == area]['pi'].tolist()
pi2url_df = area2pi2url[['pi', 'url']].drop_duplicates()
pi2url_dict = pi2url_df.set_index('pi')['url'].to_dict()


def bubble(width):
  import plotly.graph_objects as go
  # colors = ["blue", "green", "red", "orange", "purple", "gray", "brown", ]
  colors = sns.color_palette("tab10", n_colors=10).as_hex()
  cat2color_dict = dict(zip(embedding_df["category"].unique(), colors))
  categories = sorted(embedding_df["category"].unique())
  embedding_df["category_color"] = embedding_df["category"].map(cat2color_dict)
  hover_text = embedding_df["area"]

  fig = go.Figure(
    go.Scatter(
      x=embedding_df["x"],
      y=embedding_df["y"],
      mode="markers+text",
      marker=dict(
          size=embedding_df["size"],
          sizemode='area',
          sizeref=2.*max(embedding_df["size"])/(100.**2),  # scale size_max=60
          opacity=0.1,
          color=embedding_df["category_color"],
      ),
      text=embedding_df["area_campus"],
      hovertext=hover_text,
      hoverinfo="text",  # Only show hovertext
      showlegend=False,
      customdata=embedding_df[["area", "category"]].values,
    )
  ) #Figure

  # Add invisible traces for legend entries
  for cat in categories:
      fig.add_trace(
          go.Scatter(
              x=[None], y=[None],  # No actual data points
              mode="markers",
              marker=dict(color=cat2color_dict[cat], 
                          opacity=0.1,
                          size=10),
              name=cat,
              showlegend=True,
          )
      )
  fig.update_layout(
      autosize=True,
      # width=800, height=800, ## comment out for auto height
      # width=None, height=None, ## comment out for auto height
      plot_bgcolor='rgba(0,0,0,0)',
      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
      hoverlabel=dict(
          bgcolor="rgba(255,255,255,0.75)",
          bordercolor="rgba(255,255,255,0.)",
          font=dict(color="darkslategray"),
      ),
      title=dict(
          text="click bubble to see PIs",
          font={"size": 12, "color": "darkslategray"},
          x=0, xanchor="left",
          y=0.98, yanchor="top",
      ), 
      legend=dict(
        orientation="h",
        x=1, xanchor="right", xref="paper",
        y=0.96, yanchor="top", yref="container",
        bgcolor="rgba(0,0,0,0)", # Transparent background
        entrywidthmode='fraction', entrywidth=.2,
        itemclick=False, itemdoubleclick=False, # disable legend interactivity
        font=dict(color="darkslategray"),
      ),
      legend_title_text="",
  )
  return fig


app = Dash(__name__, external_stylesheets=['/assets/style.css'])

app.layout = html.Div(
  [
    dcc.Store(id="dimensions"),
    html.Div(id='dummy'),
    
    # Main container with flexbox layout
    html.Div([
      # Plot area (left side)
      html.Div([
        dcc.Graph(
          id="bubble",
          figure=bubble(1200),
          style={
            'width': '100%',
            'height': '100%',
          },
          config={
            'displayModeBar': False  # hide plotly toolbar
          },
          responsive=True,
        ),
      ], className="plot-area"),
      
      # Sidebar (right side)
      html.Div([
        html.Div(id="click-info"),
      ], className="sidebar"),
    ], className="main-row"),    
  ], style={
    'width': '100vw',
    'height': '100vh',
    'marginTop': '0px',    
    'marginLeft': '15px',   
    'boxSizing': 'border-box'  # Include margin in size calculation
  }
)

# add event listener for window resizing
app.clientside_callback(
    """
    function(trigger) {
        function dummyClick() {
            document.getElementById('dummy').click()
        };
        
        window.addEventListener('resize', dummyClick)
        return window.dash_clientside.no_update
    }
    """,
    Output("dummy", "style"),
    Input("dummy", "style")
)

# store current dimension in store
app.clientside_callback(
    """
    function updateStore(click) {
        var w = window.innerWidth;
        var h = window.innerHeight; 
        return [w, h]
    }
    """,
    Output('dimensions', 'data'),
    Input('dummy', 'n_clicks')
)

# window resize handler
@app.callback(
    Output("bubble", "figure", ),
    Input('dimensions', 'data'),
)
def change_size(dimensions):
  if dimensions is None:
    return bubble(1200)

  w, h = dimensions
  patched = Patch()
  scale_factor = w / 1200

  patched['data'][0]['marker']['size'] = bubble_size * scale_factor
  patched['data'][0]['textfont']['size'] = font_size * scale_factor

  return patched

# bubble click handler
@app.callback(
  Output("click-info", "children"),
  Input("bubble", "clickData")
)
def update_sidebar(clickData):
  if clickData is None:
    return []
  
  # Extract the clicked point data
  point = clickData['points'][0]
  
  # Get the customdata which contains [area, category]
  if 'customdata' in point:
    area = point['customdata'][0]
    category = point['customdata'][1]
    
    # Get PIs for this area
    pis = area2pis_dict.get(area, [])
    pis = sorted(pis)
    
    # Create the info display
    info_children = [
      html.P(area, style={'margin-bottom': '10px'}),
    ]
    
    if pis:
      pi_links = []
      for pi in pis:
        if pi in pi2url_dict:
          pi_links.append(
            html.A(pi, href=pi2url_dict[pi], target="_blank", 
                 style={'display': 'block', 'margin-bottom': '5px'})
          )
        else:
          pi_links.append(html.Div(pi, style={'margin-bottom': '5px'}))
      info_children.extend(pi_links)
    
    return info_children
  return


if __name__ == "__main__":
  port = int(os.environ.get('PORT', 8050))
  app.run(debug=False, host='0.0.0.0', port=port)