import pandas as pd
import random
from pathlib import Path
from sklearn.manifold import MDS


class DataProcessor:
  def __init__(self, csv_path, mds_seed=None, bubble_size=60, font_size=8):
    self.bubble_size = bubble_size
    self.font_size = font_size
    self.editable_table_exclude_cols = ['area', 'category', 'size', 'area_campus']
    self.mds_seed = random.randint(0, 10000) if mds_seed is None else mds_seed
    
    # Load and process initial data
    self.df_original = self._load_data(csv_path)
    self.df_current = self.df_original.copy()
    self.categories = sorted(self.df_original["category"].unique())
    self.embedding_df = self._compute_embedding(self.df_current)
    
  def _load_data(self, csv_path):
    """Load and prepare the initial dataframe"""
    df = pd.read_csv(Path(csv_path), index_col=["campus", "area_shortname", "area"])
    df['category'] = df.idxmax(axis=1)
    df = df.reset_index()
    df['size'] = self.bubble_size
    df['area_campus'] = df['area_shortname'] + "<br>(" + df['campus'] + ")"
    df = df.sort_values(by=['campus', 'area_shortname'])
    
    return df
  
  def _compute_embedding(self, df, mds_seed=None):
    """Compute MDS embedding from dataframe"""
    if all(col in df.columns for col in ["campus", "area_shortname", "area"]):
      df = df.set_index(["campus", "area_shortname", "area"])

    # Get score columns only
    score_columns = [col for col in df.columns if col in self.categories]
    df_scores = df[score_columns]
    df_scores = df_scores.apply(pd.to_numeric, errors='coerce')
    df_norm = df_scores.div(df_scores.sum(axis=1), axis=0)
    
    # MDS embedding
    if mds_seed is not None:
      self.mds_seed = mds_seed
    print(f"mds random seed: {self.mds_seed}")
    embedding = MDS(n_components=2, n_init=4, random_state=self.mds_seed).fit_transform(df_norm)
    
    # Create embedding dataframe
    embedding_df = pd.DataFrame(embedding, columns=["x", "y"])
    embedding_df = (embedding_df - embedding_df.min()) / (embedding_df.max() - embedding_df.min())
    embedding_df = pd.concat([
      embedding_df, 
      df.reset_index()[["area_campus", "campus", 'area', 'area_shortname', 'category', 'size']]
    ], axis=1)
    
    return embedding_df
  
  def update_from_table_data(self, table_data):
    """Update embedding from edited table data"""
    updated_df = pd.DataFrame(table_data)
    self.df_current = updated_df.copy()
    self.embedding_df = self._compute_embedding(updated_df)
    print(f"original\n{self.df_original.head()}\ncurrent\n{self.df_current.head()}")
    return self.embedding_df
  
  def update_from_mds_seed(self, mds_seed=None):
    """Update embedding using a new MDS seed"""
    if mds_seed is None:
      mds_seed = random.randint(0, 10000)
    self.embedding_df = self._compute_embedding(self.df_current, mds_seed)
    return self.embedding_df