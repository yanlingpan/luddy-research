from dash import Dash, dcc, html, Input, Output, Patch, State, dash_table, no_update
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from collections import defaultdict 

from data_processor import DataProcessor


# bubble plot data
data_dir = Path("./data")
data_processor = DataProcessor(data_dir.joinpath("area2category_score_campus.csv"), mds_seed=2971) # mds_seed need to be int
df = data_processor.df_original
embedding_df = data_processor.embedding_df
categories = data_processor.categories
editable_table_exclude_cols = data_processor.editable_table_exclude_cols
bubble_size = data_processor.bubble_size
font_size = data_processor.font_size

# extra info: area pis & links
area2pi2url = pd.read_csv(data_dir.joinpath("area2pi2url.csv"))
area2pis_dict = defaultdict(list)
for area in area2pi2url['area'].unique():
  area2pis_dict[area] = area2pi2url[area2pi2url['area'] == area]['pi'].tolist()
pi2url_df = area2pi2url[['pi', 'url']].drop_duplicates()
pi2url_dict = pi2url_df.set_index('pi')['url'].to_dict()



def bubble(width):
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
    dcc.Store(id="table-visible", data=False),  # table visibility state
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
        html.Div([
          # mds input box
          html.Div([
            dcc.Input(
              id='mds-seed-input',
              placeholder='',
              value="",
              debounce=True,
              autoComplete='off',
              autoFocus=False, spellCheck=False,
              style={
                'width': '50px',
                'padding': '8px 12px',
                'background-color': 'rgba(255, 255, 255, 0.9)',
                'border': '1px solid #ccc',
                # 'border-radius': '4px',
                'font-size': '12px'
              }
            ),
          ], 
          title="enter an integer as MDS seed",
          style={
            # 'position': 'absolute',
            # 'bottom': '10px',
            # 'left': '10px',
            'z-index': '1000',
            'margin-right': '5px'
          }),
          # Button positioned in lower left corner
          html.Button(
            "change MDS seed",
            id="mds-seed-button",
            title="change random seed for MDS embedding",
            style={
              # 'position': 'absolute',
              # 'bottom': '10px',
              # 'left': '80px',
              # 'z-index': '1000',
              'padding': '8px 12px',
              'background-color': 'rgba(255, 255, 255, 0.9)',
              'border': '1px solid #ccc',
              'border-radius': '4px',
              'cursor': 'pointer',
              'font-size': '12px',
              'margin-right': '5px'
            }
          ),
          # Error message display
          html.Div(id="input-error-message", style={'display': 'none'})
        ], className="mds-control", style={
          'position': 'absolute',
          'bottom': '10px',
          'left': '0px',
          'z-index': '1000',
          'display': 'flex',  # Use flexbox for horizontal layout
          'align-items': 'center',  # Vertical alignment
          }),
      ], className="plot-area", style={'position': 'relative'}),
      
      # Sidebar (right side)
      html.Div([
        html.Div(id="click-info"),
      ], className="sidebar"),
    ], className="main-row"),
    
    # Editable table
    html.Div([
      html.Button("edit score table", id="toggle-table-btn", n_clicks=0,),

      html.Button("download original table", id="btn-download-orig",),
      dcc.Download(id="download-curr-table",),
      
      html.Button("download current table", id="btn-download-curr",),
      dcc.Download(id="download-orig-table"),
      
      dcc.Upload(
        html.Button("upload score table (not implemented yet)", id="btn-upload",),
        id="upload-table"
      ),
    ], className="button-row", style={'display': 'flex', 'flex-wrap': 'wrap'}),
    html.Div([
      html.Div([dash_table.DataTable( # make score table editable
          id='editable-table',
          data=df.to_dict('records'),
          columns=[{"name": i, "id": i} for i in df.columns if i not in editable_table_exclude_cols],
          editable=True,
          # page_size=10,
          fixed_rows={'headers': True},
          fixed_columns={'headers': True, 'data': 2},
          style_table={
            'height': '350px',
            # 'overflowX': 'auto', 
            # 'overflowY': 'auto',
            'minWidth': '80%',
            'maxWidth': '100%'
          },
          style_data_conditional=[
            {'if': {'column_id': 'campus'},
            'width': '50px', 'textAlign': 'right'},
            {'if': {'column_id': 'area_shortname'},
            'width': '140px', 'textAlign': 'left'},
          ],
          style_header={'textAlign': 'center'},
          style_cell_conditional=[{'if': {'column_id': c},
               'textAlign': 'center',
               'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
               } for c in categories
          ],
        ),
      ], className="editable-table-container"),
      html.Div([
        html.Div(id="click-info-table"),  # Secondary click info for table view
      ], className="dummy_sidebar"),
    ], className="table-row", id="table-container", style={'display': 'none'}),
  ], style={
    'width': '100vw',
    'height': '100vh',
    'marginTop': '0px',    
    'marginLeft': '15px',   
    'boxSizing': 'border-box'  # Include margin in size calculation
  },
  
)

# download original table as CSV
@app.callback(
  Output("download-orig-table", "data"),
  Input("btn-download-orig", "n_clicks"),
  prevent_initial_call=True,
)
def func(n_clicks):
  return dcc.send_file(data_dir.joinpath("area2category_score_campus.csv"))


# download current table as CSV
@app.callback(
  Output("download-curr-table", "data"),
  Input("btn-download-curr", "n_clicks"),
  prevent_initial_call=True,
)
def func(n_clicks):
  cols2include = pd.read_csv(data_dir.joinpath("area2category_score_campus.csv")).columns.tolist()
  df_curr = data_processor.df_current[cols2include]
  return dcc.send_data_frame(df_curr.to_csv, "area2category_score_campus_current.csv", index=False)


# type in MDS seed
@app.callback(
  [Output("bubble", "figure", allow_duplicate=True),
   Output("mds-seed-input", "value", allow_duplicate=True),
   Output("input-error-message", "children"),
   Output("input-error-message", "style")],
  Input("mds-seed-input", "value"),
  State('dimensions', 'data'),
  prevent_initial_call=True
)
def type_mds_seed(mds_seed, dimensions=None):
  error_style = {
    'color': 'red',
    'font-size': '12px',
    'padding': '8px',
  }
  # Re-embed with the new seed
  global embedding_df
  try:
    mds_seed = int(float(mds_seed))
    embedding_df = data_processor.update_from_mds_seed(mds_seed)
    error_style['display'] = 'none'
    error_message = ""
  except ValueError:
    error_message = "MDS seed must be an integer"
    return no_update, no_update, error_message, error_style
  
  # Create new bubble plot with updated data
  new_figure = bubble(1200)
  if dimensions is not None:
    w, h = dimensions
    scale_factor = w / 1200
    new_figure['data'][0]['marker']['size'] = bubble_size * scale_factor
    new_figure['data'][0]['textfont']['size'] = font_size * scale_factor
  
  return new_figure, str(data_processor.mds_seed), error_message, error_style

# button random change MDS seed
@app.callback(
  [Output("bubble", "figure", allow_duplicate=True),
   Output("mds-seed-input", "value", allow_duplicate=True)],
  Input("mds-seed-button", "n_clicks"),
  State('dimensions', 'data'),
  prevent_initial_call=True
)
def change_mds_seed(n_clicks, dimensions=None):
  if n_clicks == 0:
    return no_update
  
  # Re-embed with a new seed
  global embedding_df
  embedding_df = data_processor.update_from_mds_seed()
  
  # Create new bubble plot with updated data
  new_figure = bubble(1200)
  if dimensions is not None:
    w, h = dimensions
    scale_factor = w / 1200
    new_figure['data'][0]['marker']['size'] = bubble_size * scale_factor
    new_figure['data'][0]['textfont']['size'] = font_size * scale_factor
  
  return new_figure, str(data_processor.mds_seed)


@app.callback(
  [Output('editable-table', 'data'),
   Output("bubble", "figure", allow_duplicate=True)],
  Input('editable-table', 'data_timestamp'),
  [State('editable-table', 'data'),
   State('dimensions', 'data')],
  prevent_initial_call=True
)
def update_table_and_graph(data_timestamp, current_data, dimensions=None):  
  # Re-embed with MDS using the updated data
  global embedding_df
  embedding_df = data_processor.update_from_table_data(current_data)
  
  # Create new bubble plot with updated data
  new_figure = bubble(1200)
  if dimensions is not None:
    w, h = dimensions
    scale_factor = w / 1200
    new_figure['data'][0]['marker']['size'] = bubble_size * scale_factor
    new_figure['data'][0]['textfont']['size'] = font_size * scale_factor
  
  return current_data, new_figure


# Toggle table visibility
@app.callback(
  [Output("table-container", "style"),
   Output("toggle-table-btn", "children"),
   Output("table-visible", "data")],
  Input("toggle-table-btn", "n_clicks"),
  State("table-visible", "data")
)
def toggle_table(n_clicks, is_visible):
  if n_clicks == 0:
    return {'display': 'none'}, "edit score table", False
  
  if is_visible:
    return {'display': 'none'}, "edit score table", False
  else:
    return {'display': 'flex'}, "hide score table", True


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
    Output("dummy", "style"), # !!do not allow_duplicate, do not prevent_initial_call!!
    Input("dummy", "style"),
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
    Output("bubble", "figure", allow_duplicate=True),
    Input('dimensions', 'data'),
    prevent_initial_call=True
)
def resize_graph(dimensions):
  if dimensions is None:
    return bubble(1200)

  w, h = dimensions
  patched = Patch()
  scale_factor = w / 1200
  # print(f"scale factor: {scale_factor:.2f} bubble size: {round(bubble_size*scale_factor)} font size: {round(font_size*scale_factor)}")
  # # scale marker size depending on window width (w)
  # for i in range(len(embedding_df['category'].unique())):
  #   patched['data'][i]['marker']['size'] = bubble_size * scale_factor
  #   patched['data'][i]['textfont']['size'] = font_size * scale_factor
  # Only update the main scatter trace (index 0), not the legend traces
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
    return []#[html.P("click on a bubble to see PIs...", 
            #        style={"fontSize": "14px", "color": "darkslategray"}
            #        )
            # ]
  
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
      # html.P(f"Category: {category}", style={'margin-bottom': '15px', 'font-style': 'italic'}),
    ]
    
    if pis:
      # info_children.append(html.H6("Principal Investigators:", )) #style={'margin-bottom': '10px'}
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