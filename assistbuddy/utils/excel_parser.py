
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class ExcelParser:
    """
    Parse Excel files and perform basic analytics
    """
    
    def __init__(self):
        self.df = None
        self.file_path = None
    
    def load_excel(
        self,
        file_path: str,
        sheet_name: Optional[str] = None
    ) -> pd.DataFrame:
        
        self.file_path = file_path
        
        try:
            if sheet_name:
                self.df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                self.df = pd.read_excel(file_path)
            
            return self.df
        
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        
        if self.df is None:
            return {}
        
        summary = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'total_missing': int(self.df.isnull().sum().sum()),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        
        # Numeric column stats
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = self.df[numeric_cols].describe().to_dict()
        
        return summary
    
    def detect_outliers(
        self,
        column: str,
        method: str = 'iqr',  # 'iqr' or 'zscore'
        threshold: float = 3.0
    ) -> Tuple[List[int], List[Any]]:
       
        if self.df is None or column not in self.df.columns:
            return [], []
        
        data = self.df[column].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outlier_mask = z_scores > threshold
        
        else:
            return [], []
        
        outlier_indices = data[outlier_mask].index.tolist()
        outlier_values = data[outlier_mask].tolist()
        
        return outlier_indices, outlier_values
    
    def get_pivot_summary(
        self,
        index: str,
        values: str,
        aggfunc: str = 'sum'
    ) -> pd.DataFrame:
       
        if self.df is None:
            return None
        
        try:
            pivot = pd.pivot_table(
                self.df,
                index=index,
                values=values,
                aggfunc=aggfunc
            )
            return pivot
        except Exception as e:
            print(f"Error creating pivot: {e}")
            return None
    
    def find_suspicious_rows(self) -> List[Dict]:
       
        if self.df is None:
            return []
        
        suspicious = []
        
        # Check for rows with too many missing values
        missing_threshold = len(self.df.columns) * 0.5
        for idx, row in self.df.iterrows():
            missing_count = row.isnull().sum()
            if missing_count >= missing_threshold:
                suspicious.append({
                    'row': int(idx),
                    'reason': f'{missing_count} missing values',
                    'severity': 'medium'
                })
        
        # Check numeric columns for outliers
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outlier_indices, outlier_values = self.detect_outliers(col, method='iqr')
            for idx, val in zip(outlier_indices, outlier_values):
                suspicious.append({
                    'row': int(idx),
                    'reason': f'{col} value {val} is outlier',
                    'severity': 'high'
                })
        
        return suspicious
    
    def get_text_representation(self, max_rows: int = 50) -> str:
        
        if self.df is None:
            return ""
        
        summary = self.get_summary()
        
        lines = [
            f"Excel File: {self.file_path or 'Unknown'}",
            f"Rows: {summary['rows']}, Columns: {summary['columns']}",
            f"Missing values: {summary['total_missing']}",
            "",
            "Column names: " + ", ".join(summary['column_names']),
            ""
        ]
        
        # Add sample rows
        lines.append("Sample data (first 10 rows):")
        lines.append(self.df.head(10).to_string())
        
        # Add suspicious rows if any
        suspicious = self.find_suspicious_rows()
        if suspicious:
            lines.append(f"\n⚠️ Found {len(suspicious)} suspicious rows:")
            for item in suspicious[:5]:  # Top 5
                lines.append(f"  - Row {item['row']}: {item['reason']}")
        
        return "\n".join(lines)
    
    def get_all_sheets(self, file_path: str) -> List[str]:
        """Get all sheet names in Excel file"""
        try:
            xl_file = pd.ExcelFile(file_path)
            return xl_file.sheet_names
        except Exception as e:
            print(f"Error reading sheets: {e}")
            return []


# Example usage
if __name__ == "__main__":
    parser = ExcelParser()
    
    # Load and analyze
    df = parser.load_excel("data.xlsx")
    
    if df is not None:
        summary = parser.get_summary()
        print(f"Rows: {summary['rows']}, Columns: {summary['columns']}")
        print(f"Missing: {summary['total_missing']}")
        
        # Find outliers
        if 'Sales' in df.columns:
            indices, values = parser.detect_outliers('Sales')
            print(f"\nOutliers in Sales: {len(indices)}")
        
        # Get text summary
        print("\n" + parser.get_text_representation())
