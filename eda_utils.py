import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def get_basic_info(self):
        """Get basic information about the dataset"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        }
        return info
    
    def get_summary_statistics(self):
        """Get summary statistics for numerical columns"""
        return self.df.describe().to_dict()
    
    def get_categorical_summary(self):
        """Get summary for categorical columns"""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cat_summary = {}
        for col in cat_cols:
            cat_summary[col] = {
                'unique_count': self.df[col].nunique(),
                'unique_values': self.df[col].unique().tolist()[:10],  # First 10 unique values
                'value_counts': self.df[col].value_counts().head(10).to_dict()
            }
        return cat_summary
    
    def create_histogram(self, column, bins=30):
        """Create histogram for numerical column"""
        if column not in self.df.select_dtypes(include=[np.number]).columns:
            return None
        
        fig = px.histogram(self.df, x=column, nbins=bins, title=f'Distribution of {column}')
        fig.update_layout(showlegend=False)
        return fig
    
    def create_box_plot(self, column):
        """Create box plot for numerical column"""
        if column not in self.df.select_dtypes(include=[np.number]).columns:
            return None
        
        fig = px.box(self.df, y=column, title=f'Box Plot of {column}')
        return fig
    
    def create_bar_chart(self, column, top_n=10):
        """Create bar chart for categorical column"""
        if column not in self.df.select_dtypes(include=['object', 'category']).columns:
            return None
        
        value_counts = self.df[column].value_counts().head(top_n)
        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                     title=f'Top {top_n} {column} Values',
                     labels={'x': column, 'y': 'Count'})
        fig.update_xaxes(tickangle=45)
        return fig
    
    def create_scatter_plot(self, x_col, y_col):
        """Create scatter plot between two numerical columns"""
        if x_col not in self.df.select_dtypes(include=[np.number]).columns or \
           y_col not in self.df.select_dtypes(include=[np.number]).columns:
            return None
        
        fig = px.scatter(self.df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for numerical columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = self.df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title='Correlation Heatmap')
        return fig
    
    def detect_outliers(self, column, method='iqr'):
        """Detect outliers in a numerical column"""
        if column not in self.df.select_dtypes(include=[np.number]).columns:
            return None
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(self.df) * 100,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def answer_question(self, question):
        """Enhanced rule-based question answering about the dataset"""
        question_lower = question.lower()
        
        # Basic info questions
        if 'how many rows' in question_lower or 'row count' in question_lower:
            return f"The dataset has {self.df.shape[0]} rows."
        
        if 'how many columns' in question_lower or 'column count' in question_lower:
            return f"The dataset has {self.df.shape[1]} columns."
        
        if 'columns' in question_lower and 'name' in question_lower:
            return f"The columns are: {', '.join(self.df.columns.tolist())}"
        
        # Missing values
        if 'missing' in question_lower or 'null' in question_lower:
            missing_count = self.df.isnull().sum().sum()
            return f"There are {missing_count} missing values in the dataset."
        
        # Data types
        if 'data type' in question_lower or 'dtype' in question_lower:
            dtype_info = self.df.dtypes.value_counts().to_dict()
            return f"Data types: {dtype_info}"
        
        # Enhanced questions about students, exams, pass/fail, backlogs
        if 'student' in question_lower or 'students' in question_lower:
            # Pass/Fail analysis
            if 'pass' in question_lower and 'without backlog' in question_lower:
                for col in self.df.columns:
                    if 'pass' in col.lower() or 'result' in col.lower() or 'status' in col.lower():
                        pass_data = self.df[col].value_counts()
                        pass_count = 0
                        for val, count in pass_data.items():
                            if 'pass' in str(val).lower() or 'clear' in str(val).lower():
                                pass_count += count
                        return f"Based on the {col} column, {pass_count} students passed ({pass_count/len(self.df)*100:.1f}% of total students)."
            
            if 'pass' in question_lower:
                for col in self.df.columns:
                    if 'pass' in col.lower() or 'result' in col.lower() or 'status' in col.lower():
                        pass_data = self.df[col].value_counts()
                        return f"Pass/Fail distribution from {col} column: {dict(pass_data)}"
            
            if 'backlog' in question_lower:
                for col in self.df.columns:
                    if 'backlog' in col.lower():
                        backlog_data = self.df[col].value_counts()
                        no_backlog = 0
                        for val, count in backlog_data.items():
                            if 'no' in str(val).lower() or '0' in str(val) or 'clear' in str(val).lower():
                                no_backlog += count
                        return f"Students without backlogs: {no_backlog} ({no_backlog/len(self.df)*100:.1f}%). Full backlog data: {dict(backlog_data)}"
            
            # Academic performance
            if 'academic' in question_lower or 'performance' in question_lower:
                for col in self.df.columns:
                    if 'grade' in col.lower() or 'score' in col.lower() or 'marks' in col.lower() or 'percentage' in col.lower():
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            avg_score = self.df[col].mean()
                            max_score = self.df[col].max()
                            min_score = self.df[col].min()
                            return f"Academic performance from {col}: Average={avg_score:.2f}, Max={max_score:.2f}, Min={min_score:.2f}"
            
            # Placement questions
            if 'placement' in question_lower:
                for col in self.df.columns:
                    if 'placement' in col.lower() or 'placed' in col.lower():
                        placement_data = self.df[col].value_counts()
                        placed_count = 0
                        for val, count in placement_data.items():
                            if 'yes' in str(val).lower() or 'placed' in str(val).lower() or '1' in str(val):
                                placed_count += count
                        return f"Students placed: {placed_count} ({placed_count/len(self.df)*100:.1f}%). Full placement data: {dict(placement_data)}"
        
        # Column-specific questions
        for col in self.df.columns:
            if col.lower() in question_lower:
                if 'mean' in question_lower or 'average' in question_lower:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        return f"The mean of {col} is {self.df[col].mean():.2f}"
                
                if 'max' in question_lower or 'maximum' in question_lower:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        return f"The maximum value of {col} is {self.df[col].max()}"
                
                if 'min' in question_lower or 'minimum' in question_lower:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        return f"The minimum value of {col} is {self.df[col].min()}"
                
                if 'unique' in question_lower:
                    return f"The column {col} has {self.df[col].nunique()} unique values"
                
                # Value counts for categorical columns
                if pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                    value_counts = self.df[col].value_counts()
                    return f"Distribution of {col}: {dict(value_counts.head(10))}"
        
        # If no specific match, provide helpful suggestions
        return f"I can help you analyze your student data! Try asking about:\n• Pass/fail rates\n• Backlog statistics\n• Academic performance\n• Placement rates\n• Specific column statistics\n\nAvailable columns: {', '.join(self.df.columns.tolist())}"
