import pandas as pd


class Item:
  def __init__(self, items: list, support: float):
    self.items = set(items)
    self.support = support

  def __str__(self):
    item_str = ', '.join(sorted(self.items))
    return f"Items: [{item_str}], Support: {self.support:.2f}"

  def __eq__(self, other):
    if isinstance(other, self.__class__):
        return self.items == other.items
    else:
        return False
  
  def __ne__(self, other):
    return not self.__eq__(other)


class Apriori:
  def __init__(self, df: pd.DataFrame, 
               attributes: list, 
               class_name: str,
               min_support: float,
               max_stoppage: int = 15) -> None:
    """
    Generates the association rules

    Parameters:
      df: pd.DataFrame. The dataset
      attributes: list. List of variable names
      class_name: str. The name of the class
      min_support: float. Minimum support
      max_stoppage: int. Max k-itemsets to stop at


    Note: Assuming categorical variables are of form 0 to 1
    Returns:
      None
    """
    self.df = df
    self.attributes = attributes
    self.class_name = class_name
    self.min_support = min_support
    self.max_stoppage = max_stoppage

    self.attribute_df = self.df.drop([self.class_name], axis=1)
    self.columns = self.attribute_df.columns
    self.class_df = self.df[class_name]
    
  def generate(self):
    self._generate_1_itemset()
    self._generate_2_itemset()
    
    for k in range(3, self.max_stoppage):
      print(k)
      self._generate_k_itemset(k)
      if len(self.itemset[k]) == 0:
        break
    
  def _generate_1_itemset(self):
    """
    Generate 1 itemset

    Parameters:
      None
      
    Returns:
      items: dict of key, value pairs with the key the variable name and the value the support
    """
    k = 1
    self.itemset = {}
    self.itemset[k] = []
    self.initial_items = []
    num_rows = self.df.shape[0]
    for column in self.columns:
      support = sum(self.attribute_df[column] == 1) / num_rows
      if support >= self.min_support:
        item = Item([column], support)
        self.itemset[k].append(item)
        self.initial_items.append(column)

  def _generate_2_itemset(self):
    columns = self.initial_items
    k = 2
    self.itemset[k] = []
    num_rows = self.df.shape[0]
    for i in range(len(columns)):
      for j in range(i + 1, len(columns)):
        column1_name = columns[i]
        column2_name = columns[j]
        # Construct the query expression
        query_expression = f"{column1_name} == 1 & {column2_name} == 1"
        support = len(self.attribute_df.query(query_expression)) / num_rows
        if support >= self.min_support:
          item = Item([column1_name, column2_name], support)
          self.itemset[k].append(item)

  def _generate_k_itemset(self, k):
    prior_columns = self.itemset[k - 1]
    initial_columns = self.initial_items
    self.itemset[k] = []
    num_rows = self.df.shape[0]
    
    for i in range(len(prior_columns)):
      for j in range(len(initial_columns)):
        # Check 1: Can't have same variable_name in current k - 1 rule check
        if initial_columns[j] not in prior_columns[i].items:
          # Union the two
          new_item = prior_columns[i].items | {initial_columns[j]}
          # Check 2: Can't have duplicate different ordered rules
            # ie. {A, B} and {B, A} are the same
            # check if new_item is in the itemset regardless of order
          if not any(frozenset(new_item) == frozenset(item.items) for item in self.itemset[k]):
            # Construct the query expression
            query_expression = ""
            for item in new_item:
              query_expression = query_expression + item + " == 1 & "
            query_expression = query_expression[:-2]
            support = len(self.attribute_df.query(query_expression)) / num_rows
            if support >= self.min_support:
              item = Item(new_item, support)
              self.itemset[k].append(item)
            #self.itemset[k].append(new_item)