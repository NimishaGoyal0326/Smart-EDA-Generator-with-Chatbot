"""
Smart EDA Engine - Self-trained ML models for CSV analysis.
Replaces OpenAI API + RAG with local scikit-learn models.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, r2_score,
                             mean_squared_error, mean_absolute_error,
                             roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')


class SmartEDA:
    """Self-contained EDA + ML engine. No external APIs needed."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.models = {}
        self.label_encoders = {}
        self.scaler = None
        self.target_col = None
        self.task_type = None  # 'classification' or 'regression'
        self.feature_cols = []
        self.training_report = None
        self.full_eda_cache = None

    # ─── TARGET DETECTION ───────────────────────────────────────
    def detect_target(self):
        """Auto-detect the most likely target column."""
        keywords = ['target', 'label', 'class', 'status', 'default', 'outcome',
                     'result', 'placement', 'placed', 'survived', 'churn', 'fraud',
                     'diagnosis', 'y', 'output', 'pass', 'fail']
        for col in self.df.columns:
            if col.lower().strip() in keywords:
                return col
        for col in self.df.columns:
            for kw in keywords:
                if kw in col.lower():
                    return col
        # Last column with few unique values
        last_col = self.df.columns[-1]
        if self.df[last_col].nunique() <= 20:
            return last_col
        return None

    def set_target(self, col_name: str):
        self.target_col = col_name
        nunique = self.df[col_name].nunique()
        if pd.api.types.is_numeric_dtype(self.df[col_name]) and nunique > 15:
            self.task_type = 'regression'
        else:
            self.task_type = 'classification'

    # ─── PREPROCESSING ──────────────────────────────────────────
    def _prepare_data(self):
        """Prepare features and target for ML training."""
        if self.target_col is None:
            return None, None
        df_ml = self.df.dropna(subset=[self.target_col]).copy()
        y = df_ml[self.target_col].copy()
        X = df_ml.drop(columns=[self.target_col])

        # Drop columns with too many missing or unique text values
        drop_cols = []
        for c in X.columns:
            if X[c].isnull().sum() / len(X) > 0.5:
                drop_cols.append(c)
            elif pd.api.types.is_object_dtype(X[c]) and X[c].nunique() > 50:
                drop_cols.append(c)
        X = X.drop(columns=drop_cols)

        # Encode categoricals
        self.label_encoders = {}
        for c in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            self.label_encoders[c] = le

        # Encode target if classification
        if self.task_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=self.target_col)
            self.label_encoders['__target__'] = le

        # Fill remaining NaN
        X = X.fillna(X.median(numeric_only=True))
        for c in X.select_dtypes(include=['object', 'category']).columns:
            X[c] = X[c].fillna(X[c].mode()[0] if len(X[c].mode()) > 0 else 0)

        # Keep only numeric
        X = X.select_dtypes(include=[np.number])
        self.feature_cols = X.columns.tolist()

        # Scale
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        return X_scaled, y

    # ─── MODEL TRAINING ─────────────────────────────────────────
    def train_models(self):
        """Train multiple ML models and return a report dict."""
        X, y = self._prepare_data()
        if X is None or len(X) < 10:
            return {"error": "Not enough data to train models."}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        if self.task_type == 'classification':
            candidates = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
        else:
            candidates = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            }

        results = {}
        best_score = -1e9
        best_name = None

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if self.task_type == 'classification':
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_test, y_pred)
                    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    entry = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                             'confusion_matrix': cm, 'classification_report': cls_report}
                    # ROC (binary or OVR)
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)
                            unique_classes = np.unique(y_test)
                            if len(unique_classes) == 2:
                                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                                auc_val = roc_auc_score(y_test, y_proba[:, 1])
                                entry['roc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc_val}
                            else:
                                auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                                entry['roc'] = {'auc': auc_val}
                    except Exception:
                        pass
                    results[name] = entry
                    score = acc
                else:
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    results[name] = {'r2': r2, 'mse': mse, 'mae': mae}
                    score = r2
                self.models[name] = model
                if score > best_score:
                    best_score = score
                    best_name = name
            except Exception as e:
                results[name] = {'error': str(e)}

        # Feature importance from best tree model
        feat_imp = {}
        for n in ['Random Forest', 'Gradient Boosting']:
            if n in self.models and hasattr(self.models[n], 'feature_importances_'):
                imp = self.models[n].feature_importances_
                feat_imp = dict(sorted(zip(self.feature_cols, imp), key=lambda x: x[1], reverse=True))
                break

        # Get class labels for confusion matrix display
        class_labels = None
        if self.task_type == 'classification':
            if '__target__' in self.label_encoders:
                class_labels = self.label_encoders['__target__'].classes_.tolist()
            else:
                class_labels = sorted([str(c) for c in np.unique(y)])

        self.training_report = {
            'task_type': self.task_type,
            'target': self.target_col,
            'features_used': self.feature_cols,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'results': results,
            'best_model': best_name,
            'best_score': best_score,
            'feature_importance': feat_imp,
            'class_labels': class_labels,
        }
        return self.training_report

    # ─── PREDICTION ──────────────────────────────────────────────
    def predict_single(self, input_dict: dict, model_name: str = None):
        """Predict for a single row given as dict."""
        if not self.models:
            return "No models trained yet. Please train first."
        if model_name is None:
            model_name = self.training_report.get('best_model') if self.training_report else list(self.models.keys())[0]
        model = self.models.get(model_name)
        if model is None:
            return f"Model '{model_name}' not found."

        row = pd.DataFrame([input_dict])
        for c in self.feature_cols:
            if c not in row.columns:
                row[c] = 0
        for c, le in self.label_encoders.items():
            if c in row.columns and c != '__target__':
                try:
                    row[c] = le.transform(row[c].astype(str))
                except Exception:
                    row[c] = 0
        row = row[self.feature_cols].fillna(0)
        row_scaled = self.scaler.transform(row)
        pred = model.predict(row_scaled)[0]
        if '__target__' in self.label_encoders:
            pred = self.label_encoders['__target__'].inverse_transform([int(pred)])[0]
        return pred

    # ─── FULL EDA ────────────────────────────────────────────────
    def run_full_eda(self):
        """Run comprehensive EDA and cache results."""
        df = self.df
        report = {}

        # 1. Overview
        report['overview'] = {
            'rows': df.shape[0], 'columns': df.shape[1],
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicates': int(df.duplicated().sum()),
            'column_names': df.columns.tolist(),
        }

        # 2. Data types
        report['dtypes'] = {}
        for c in df.columns:
            report['dtypes'][c] = str(df[c].dtype)

        # 3. Missing values
        mv = df.isnull().sum()
        report['missing'] = {c: {'count': int(v), 'pct': round(v / len(df) * 100, 2)} for c, v in mv.items() if v > 0}
        report['total_missing'] = int(mv.sum())

        # 4. Numeric stats
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        report['numeric_stats'] = {}
        for c in num_cols:
            col = df[c].dropna()
            report['numeric_stats'][c] = {
                'mean': round(col.mean(), 4), 'median': round(col.median(), 4),
                'std': round(col.std(), 4), 'min': round(col.min(), 4), 'max': round(col.max(), 4),
                'q25': round(col.quantile(0.25), 4), 'q75': round(col.quantile(0.75), 4),
                'skew': round(col.skew(), 4), 'kurtosis': round(col.kurtosis(), 4),
                'zeros': int((col == 0).sum()), 'negatives': int((col < 0).sum()),
            }
            try:
                mode_val = col.mode()
                report['numeric_stats'][c]['mode'] = round(float(mode_val.iloc[0]), 4) if len(mode_val) > 0 else None
            except Exception:
                report['numeric_stats'][c]['mode'] = None

        # 5. Categorical stats
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        report['categorical_stats'] = {}
        for c in cat_cols:
            vc = df[c].value_counts()
            report['categorical_stats'][c] = {
                'unique': int(df[c].nunique()),
                'top_values': {str(k): int(v) for k, v in vc.head(10).items()},
                'mode': str(vc.index[0]) if len(vc) > 0 else None,
            }

        # 6. Correlations
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            high = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.5:
                        high.append({'col1': num_cols[i], 'col2': num_cols[j], 'corr': round(val, 4)})
            high.sort(key=lambda x: abs(x['corr']), reverse=True)
            report['high_correlations'] = high

        # 7. Outliers (IQR)
        report['outliers'] = {}
        for c in num_cols:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_out = int(((df[c] < lower) | (df[c] > upper)).sum())
            if n_out > 0:
                report['outliers'][c] = {'count': n_out, 'pct': round(n_out / len(df) * 100, 2),
                                         'lower_bound': round(lower, 4), 'upper_bound': round(upper, 4)}

        self.full_eda_cache = report
        return report

    # ─── QUERY ANSWERING ─────────────────────────────────────────
    def answer_question(self, question: str) -> str:
        """Answer any data question using computed analysis. No API calls."""
        q = question.lower().strip()

        # Ensure EDA is cached
        if self.full_eda_cache is None:
            self.run_full_eda()
        eda = self.full_eda_cache

        # ── Shape / overview
        if any(w in q for w in ['how many rows', 'row count', 'number of rows', 'total rows',
                                'shape', 'size of', 'total number of records', 'number of records']):
            o = eda['overview']
            return (f"**Dataset Shape:** {o['rows']} rows × {o['columns']} columns\n\n"
                    f"**Memory:** {o['memory_mb']} MB\n**Duplicates:** {o['duplicates']}")

        if any(w in q for w in ['how many columns', 'column count', 'number of columns',
                                'list columns', 'column names', 'what columns',
                                'number of features', 'how many features']):
            cols = eda['overview']['column_names']
            return f"**{len(cols)} Columns:**\n" + "\n".join(f"- `{c}` ({eda['dtypes'][c]})" for c in cols)

        # ── Missing values
        if 'missing' in q or 'null' in q or 'nan' in q:
            if not eda['missing']:
                return "✅ **No missing values** in the dataset."
            lines = [f"**Missing Values (Total: {eda['total_missing']}):**\n"]
            for c, info in sorted(eda['missing'].items(), key=lambda x: x[1]['count'], reverse=True):
                lines.append(f"- `{c}`: {info['count']} ({info['pct']}%)")
            return "\n".join(lines)

        # ── Duplicates
        if 'duplicate' in q:
            d = eda['overview']['duplicates']
            return (f"**Duplicate Rows:** {d} ({round(d / eda['overview']['rows'] * 100, 2)}%)"
                    if d else "✅ No duplicate rows found.")

        # ── Advisory / recommendation questions (BEFORE prediction)
        if self._is_advisory_question(q):
            return self._handle_advisory_question(q)

        # ── Prediction questions
        if any(w in q for w in ['predict', 'prediction', 'forecast', 'will it', 'will be',
                                'tomorrow', 'next', 'future', 'expect', 'going to']):
            return self._handle_prediction_question(q)

        # ── Target distribution / percentage / imbalance
        if self._is_target_distribution_question(q):
            return self._handle_target_distribution(q)

        # ── Group-by comparison questions (X of defaulters vs non-defaulters)
        if self._is_groupby_question(q):
            return self._handle_groupby_comparison(q)

        # ── Correlation with target
        if ('correlation' in q or 'correlated' in q) and self._mentions_target(q):
            return self._handle_target_correlation()

        # ── How many / count questions (smart counting)
        if 'how many' in q or 'count of' in q or 'number of' in q or 'total' in q:
            result = self._handle_count_question(q)
            if result:
                return result

        # ── Summary / statistics / describe
        if any(w in q for w in ['summary', 'statistics', 'describe', 'stats']):
            return self._format_numeric_stats(eda)

        # ── Correlation (general)
        if 'correlation' in q or 'correlated' in q:
            return self._format_correlations(eda)

        # ── Outlier
        if 'outlier' in q:
            return self._format_outliers(eda)

        # ── Distribution / skew
        if 'distribution' in q or 'skew' in q or 'kurtosis' in q:
            if self._mentions_target(q):
                return self._handle_target_distribution(q)
            return self._format_distributions(eda)

        # ── Probability / chance
        if any(w in q for w in ['probability', 'chance', 'likelihood', 'odds']):
            return self._handle_probability(q)

        # ── Model / accuracy / training results
        if any(w in q for w in ['accuracy', 'precision', 'recall', 'f1',
                                'training result', 'best model']):
            return self._format_training_report()

        # ── Feature importance
        if any(w in q for w in ['feature importance', 'important feature', 'most important',
                                'key factor', 'suggest feature', 'important for predict']):
            return self._format_feature_importance()

        # ── Unique values / categories / frequent
        if any(w in q for w in ['unique', 'category', 'categories', 'frequent',
                                'most common', 'value count']):
            return self._format_categorical(eda, q)

        # ── Data types
        if 'data type' in q or 'dtype' in q or 'type of' in q:
            lines = ["**Column Data Types:**\n"]
            for c, t in eda['dtypes'].items():
                lines.append(f"- `{c}`: {t}")
            return "\n".join(lines)

        # ── Column-specific: mean/max/min/average
        return self._handle_column_query(q, eda)

    # ─── FORMATTING HELPERS ──────────────────────────────────────
    def _format_numeric_stats(self, eda):
        if not eda['numeric_stats']:
            return "No numeric columns found."
        lines = ["**Summary Statistics:**\n"]
        for c, s in eda['numeric_stats'].items():
            lines.append(f"**`{c}`**: Mean={s['mean']}, Median={s['median']}, Std={s['std']}, "
                         f"Min={s['min']}, Max={s['max']}, Skew={s['skew']}")
        return "\n".join(lines)

    def _format_correlations(self, eda):
        hc = eda.get('high_correlations', [])
        if not hc:
            return "No strong correlations (>0.5) found between numeric columns."
        lines = ["**High Correlations (|r| > 0.5):**\n"]
        for item in hc:
            emoji = "🔴" if abs(item['corr']) > 0.8 else "🟡"
            lines.append(f"{emoji} `{item['col1']}` ↔ `{item['col2']}`: **{item['corr']}**")
        return "\n".join(lines)

    def _format_outliers(self, eda):
        if not eda['outliers']:
            return "✅ No outliers detected using IQR method."
        lines = ["**Outliers (IQR Method):**\n"]
        for c, info in sorted(eda['outliers'].items(), key=lambda x: x[1]['count'], reverse=True):
            lines.append(f"- `{c}`: {info['count']} outliers ({info['pct']}%) | Bounds: [{info['lower_bound']}, {info['upper_bound']}]")
        return "\n".join(lines)

    def _format_distributions(self, eda):
        if not eda['numeric_stats']:
            return "No numeric columns for distribution analysis."
        lines = ["**Distribution Analysis:**\n"]
        for c, s in eda['numeric_stats'].items():
            skew = s['skew']
            if abs(skew) > 2:
                sk_label = "Highly skewed"
            elif abs(skew) > 1:
                sk_label = "Moderately skewed"
            else:
                sk_label = "~Normal"
            direction = "right" if skew > 0 else "left" if skew < 0 else ""
            lines.append(f"- `{c}`: {sk_label} {direction} (skew={skew}, kurtosis={s['kurtosis']})")
        return "\n".join(lines)

    def _format_categorical(self, eda, q):
        cs = eda.get('categorical_stats', {})
        if not cs:
            return "No categorical columns found."
        lines = ["**Categorical Column Summary:**\n"]
        for c, info in cs.items():
            top = list(info['top_values'].items())[:5]
            top_str = ", ".join(f"'{k}': {v}" for k, v in top)
            lines.append(f"**`{c}`** ({info['unique']} unique) — Top: {top_str}")
        return "\n".join(lines)

    def _format_training_report(self):
        if not self.training_report:
            return "⚠️ No models trained yet. Click **Train Models** in the sidebar to train."
        r = self.training_report
        lines = [f"**Model Training Report**\n",
                 f"**Task:** {r['task_type'].title()} on `{r['target']}`",
                 f"**Train/Test Split:** {r['train_size']}/{r['test_size']} samples",
                 f"**Features Used:** {len(r['features_used'])}\n"]
        for name, metrics in r['results'].items():
            if 'error' in metrics:
                lines.append(f"❌ **{name}**: {metrics['error']}")
            elif r['task_type'] == 'classification':
                lines.append(f"{'✅' if name == r['best_model'] else '▪️'} **{name}**: "
                             f"Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
                             f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
            else:
                lines.append(f"{'✅' if name == r['best_model'] else '▪️'} **{name}**: "
                             f"R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['mse']**0.5:.4f}")
        lines.append(f"\n🏆 **Best Model:** {r['best_model']} (score: {r['best_score']:.4f})")
        return "\n".join(lines)

    def _format_feature_importance(self):
        if not self.training_report or not self.training_report.get('feature_importance'):
            return "⚠️ Train models first to see feature importance."
        fi = self.training_report['feature_importance']
        lines = ["**Feature Importance (Top Features):**\n"]
        for i, (feat, imp) in enumerate(list(fi.items())[:15], 1):
            bar = "█" * int(imp * 50)
            lines.append(f"{i}. `{feat}`: {imp:.4f} {bar}")
        return "\n".join(lines)

    @staticmethod
    def _fuzzy_match(word, text):
        """Check if word is a fuzzy substring of any word in text or vice-versa.
        Returns a score (0 = no match, higher = better match)."""
        if len(word) < 3:
            return 0
        # Exact word present in text
        if word in text:
            return len(word) * 3
        # Any word in text starts with / contains this word
        for tw in text.split():
            if len(tw) < 3:
                continue
            if tw.startswith(word) or word.startswith(tw):
                overlap = min(len(tw), len(word))
                return overlap * 2
            if tw in word or word in tw:
                overlap = min(len(tw), len(word))
                return overlap
        return 0

    def _handle_count_question(self, q):
        """Handle 'how many X' questions by searching column values."""
        df = self.df
        total = len(df)
        q_words = q.split()
        negate = any(w in q for w in ['not', "don't", 'doesnt', 'does not', "didn't",
                                       'did not', 'without', 'non', 'un', 'reject', 'denied'])

        # Score each categorical column for relevance to the question
        col_scores = []
        for col in df.columns:
            if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                continue
            col_lower = col.lower().replace('_', ' ')
            col_parts = col.lower().replace('_', ' ').split()
            score = 0
            for cp in col_parts:
                for qw in q_words:
                    score += self._fuzzy_match(cp, qw)
                    score += self._fuzzy_match(qw, cp)
            # Bonus: full column name (without underscores) as substring of question
            if col_lower in q:
                score += 50
            if score > 0:
                col_scores.append((col, score))

        # Sort by score descending — best matching column first
        col_scores.sort(key=lambda x: x[1], reverse=True)

        best_col = None
        best_val = None
        best_count = None

        for col, _score in col_scores:
            vc = df[col].value_counts()

            # Pass 1: direct value match in question
            for val in vc.index:
                val_str = str(val).lower()
                if val_str in q:
                    if negate:
                        best_col, best_val, best_count = col, f"NOT {val}", total - vc[val]
                    else:
                        best_col, best_val, best_count = col, val, vc[val]
                    break

            if best_col:
                break

            # Pass 2: fuzzy value match
            for val in vc.index:
                val_str = str(val).lower()
                for vw in val_str.replace('_', ' ').split():
                    match_score = 0
                    for qw in q_words:
                        match_score += self._fuzzy_match(vw, qw)
                        match_score += self._fuzzy_match(qw, vw)
                    if match_score > 4:
                        if negate:
                            best_col, best_val, best_count = col, f"NOT {val}", total - vc[val]
                        else:
                            best_col, best_val, best_count = col, val, vc[val]
                        break
                if best_col:
                    break

            # Pass 3: if column matched well but no value matched, show whole column
            if best_col is None and _score >= 8:
                vc = df[col].value_counts()
                dist = "\n".join(f"- **{v}**: {c:,} ({c/total*100:.1f}%)" for v, c in vc.items())
                return (f"**Distribution of `{col}`** ({total:,} total records):\n\n{dist}")

            if best_col:
                break

        if best_col and best_count is not None:
            pct = best_count / total * 100
            vc = df[best_col].value_counts()
            dist = "\n".join(f"- **{v}**: {c:,} ({c/total*100:.1f}%)" for v, c in vc.items())
            return (f"**{best_count:,}** records ({pct:.1f}%) where `{best_col}` = **{best_val}** "
                    f"(out of {total:,} total)\n\n"
                    f"**Full distribution of `{best_col}`:**\n{dist}")

        # Fallback: show all categorical distributions
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            lines = ["I couldn't find an exact match. Here are the categorical distributions:\n"]
            for col in cat_cols[:5]:
                vc = df[col].value_counts()
                vals = ", ".join(f"{v}: {c:,}" for v, c in vc.head(5).items())
                lines.append(f"**`{col}`:** {vals}")
            return "\n".join(lines)
        return None

    def _handle_prediction_question(self, q):
        """Use trained model to predict based on data patterns described in the question."""
        if not self.models:
            # Auto-train if not trained yet
            target = self.target_col or self.detect_target()
            if target:
                self.set_target(target)
                report = self.train_models()
                if 'error' in report:
                    return f"⚠️ Could not auto-train: {report['error']}. Please train models from the sidebar."
            else:
                return "⚠️ No target column detected. Please select a target and train models from the sidebar."

        r = self.training_report
        best_name = r['best_model']
        model = self.models[best_name]
        target = self.target_col
        df = self.df

        # Try to extract conditions from the question to build a prediction scenario
        # Look for column values mentioned in the question
        scenario = {}
        for col in self.feature_cols:
            col_lower = col.lower().replace('_', ' ')
            for word in col_lower.split():
                if len(word) > 2 and word in q:
                    # Use the most recent / common value pattern
                    if pd.api.types.is_numeric_dtype(df[col]):
                        scenario[col] = float(df[col].median())
                    else:
                        scenario[col] = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 0
                    break

        # If no specific scenario, use the last N rows as context (recent patterns)
        if not scenario:
            # Use median of all features as baseline
            for col in self.feature_cols:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        scenario[col] = float(df[col].iloc[-min(30, len(df)):].median())
                    else:
                        recent = df[col].iloc[-min(30, len(df)):]
                        scenario[col] = recent.mode().iloc[0] if len(recent.mode()) > 0 else 0
                else:
                    scenario[col] = 0

        # Make prediction
        pred = self.predict_single(scenario)

        # Analyze confidence using historical data patterns
        lines = [f"**🔮 Prediction Result (using {best_name}):**\n"]
        lines.append(f"**Predicted `{target}`:** **{pred}**\n")

        # Show historical pattern for this prediction
        if target in df.columns:
            vc = df[target].value_counts()
            total = len(df)
            lines.append("**Historical Pattern:**")
            for val, count in vc.items():
                pct = count / total * 100
                lines.append(f"- {target} = {val}: {count:,} times ({pct:.1f}%)")

        # Show key factors
        fi = r.get('feature_importance', {})
        if fi:
            top_feats = list(fi.items())[:5]
            lines.append("\n**Top factors influencing prediction:**")
            for feat, imp in top_feats:
                val = scenario.get(feat, 'N/A')
                lines.append(f"- `{feat}` = {val} (importance: {imp:.3f})")

        # Model performance note
        best_metrics = r['results'].get(best_name, {})
        if r['task_type'] == 'classification' and 'accuracy' in best_metrics:
            lines.append(f"\n**Model Accuracy:** {best_metrics['accuracy']*100:.1f}%")
        elif r['task_type'] == 'regression' and 'r2' in best_metrics:
            lines.append(f"\n**Model R² Score:** {best_metrics['r2']:.4f}")

        lines.append("\n*Prediction based on patterns learned from the entire dataset.*")
        return "\n".join(lines)

    def _handle_probability(self, q):
        """Compute probability/chance from actual data."""
        target = self.target_col or self._find_likely_target()
        if target is None:
            return self._general_probability(q)

        col = self.df[target]
        vc = col.value_counts()
        total = len(col)
        lines = [f"**Probability Analysis for `{target}`:**\n"]
        for val, count in vc.items():
            lines.append(f"- P({target}={val}) = **{count / total * 100:.2f}%** ({count}/{total})")

        # Conditional probabilities by numeric columns
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cond_lines = []
        for c in num_cols[:5]:
            if c == target:
                continue
            med = self.df[c].median()
            above = self.df[self.df[c] >= med]
            below = self.df[self.df[c] < med]
            for val in vc.index[:2]:
                rate_above = (above[target] == val).mean() * 100
                rate_below = (below[target] == val).mean() * 100
                if abs(rate_above - rate_below) > 3:
                    cond_lines.append(f"- When `{c}` ≥ {med:.1f}: P({target}={val})={rate_above:.1f}% vs <{med:.1f}: {rate_below:.1f}%")
        if cond_lines:
            lines.append("\n**Key Conditional Probabilities:**")
            lines.extend(cond_lines[:10])
        return "\n".join(lines)

    def _general_probability(self, q):
        lines = ["**General Statistical Probabilities:**\n"]
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for c in cat_cols[:3]:
            vc = self.df[c].value_counts()
            total = len(self.df)
            probs = [f"P({c}={v})={count / total * 100:.1f}%" for v, count in vc.head(5).items()]
            lines.append(f"**`{c}`:** " + ", ".join(probs))
        return "\n".join(lines) if len(lines) > 1 else "No categorical columns for probability analysis."

    def _find_likely_target(self):
        for c in self.df.columns:
            if self.df[c].nunique() <= 5 and self.df[c].nunique() >= 2:
                return c
        return None

    # ─── TARGET-AWARE HELPERS ────────────────────────────────────
    def _get_target_col(self):
        """Return current target column or auto-detect one."""
        if self.target_col:
            return self.target_col
        return self.detect_target()

    def _mentions_target(self, q):
        """Check if the question references the target variable or related concepts."""
        target = self._get_target_col()
        if target and target.lower() in q:
            return True
        target_keywords = ['target', 'default', 'status', 'label', 'class',
                           'outcome', 'churn', 'fraud', 'survived', 'placed',
                           'result', 'dependent variable', 'output variable']
        return any(kw in q for kw in target_keywords)

    def _is_target_distribution_question(self, q):
        """Detect questions about target distribution, imbalance, or class percentages."""
        target_triggers = ['imbalance', 'imbalanced', 'balanced', 'balance',
                           'percentage of', 'percent of', 'proportion',
                           'ratio of', 'class distribution', 'target distribution',
                           'target variable', 'defaulted vs', 'vs non',
                           'non-default', 'default rate', 'how many default']
        if any(t in q for t in target_triggers):
            return True
        if ('distribution' in q) and self._mentions_target(q):
            return True
        return False

    def _handle_target_distribution(self, q):
        """Show distribution of the target variable with imbalance analysis."""
        target = self._get_target_col()
        if target is None:
            return "⚠️ No target column detected. Please select one from the sidebar."
        df = self.df
        vc = df[target].value_counts()
        total = len(df)
        lines = [f"**Distribution of Target Variable `{target}`:**\n"]
        for val, count in vc.items():
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            lines.append(f"- **{val}**: {count:,} ({pct:.2f}%) {bar}")

        # Imbalance analysis
        if len(vc) >= 2:
            majority = vc.iloc[0]
            minority = vc.iloc[-1]
            ratio = majority / minority if minority > 0 else float('inf')
            lines.append(f"\n**Imbalance Ratio:** {ratio:.2f}:1 (majority/minority)")
            if ratio > 3:
                lines.append("⚠️ **Dataset is significantly imbalanced.** "
                             "Consider SMOTE, class weights, or undersampling.")
            elif ratio > 1.5:
                lines.append("🟡 **Moderate imbalance detected.** "
                             "Consider using class_weight='balanced' in models.")
            else:
                lines.append("✅ **Dataset is reasonably balanced.**")
        return "\n".join(lines)

    def _is_groupby_question(self, q):
        """Detect comparison questions like 'average income of defaulters vs non-defaulters'."""
        comparison_signals = ['vs', 'versus', 'compared to', 'between',
                              'differ', 'vary', 'defaulter', 'non-defaulter',
                              'group by', 'grouped by', 'across']
        if any(s in q for s in comparison_signals):
            # Must also mention a feature or aggregation
            agg_words = ['average', 'mean', 'median', 'how does', 'how do',
                         'vary', 'differ', 'compare', 'income', 'loan', 'amount']
            return any(w in q for w in agg_words)
        return False

    def _handle_groupby_comparison(self, q):
        """Handle comparison questions: aggregate a feature grouped by target."""
        target = self._get_target_col()
        if target is None:
            return "⚠️ No target column detected. Please select one from the sidebar."
        df = self.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)

        # Find which numeric column the user is asking about
        matched_col = None
        for c in num_cols:
            if c.lower() in q:
                matched_col = c
                break
        if matched_col is None:
            for c in num_cols:
                for word in c.lower().replace('_', ' ').split():
                    if len(word) > 2 and word in q:
                        matched_col = c
                        break
                if matched_col:
                    break
        if matched_col is None:
            matched_col = num_cols[0] if num_cols else None
        if matched_col is None:
            return "No numeric columns available for comparison."

        grouped = df.groupby(target)[matched_col].agg(['mean', 'median', 'std', 'count'])
        lines = [f"**`{matched_col}` grouped by `{target}`:**\n"]
        for val, row in grouped.iterrows():
            lines.append(f"- **{target}={val}**: Mean={row['mean']:.2f}, "
                         f"Median={row['median']:.2f}, Std={row['std']:.2f}, "
                         f"Count={int(row['count']):,}")

        # Statistical significance hint
        groups = [g[matched_col].dropna() for _, g in df.groupby(target)]
        if len(groups) == 2 and len(groups[0]) > 1 and len(groups[1]) > 1:
            try:
                t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                sig = "statistically significant" if p_val < 0.05 else "not statistically significant"
                lines.append(f"\n**T-test:** t={t_stat:.4f}, p={p_val:.4f} → Difference is **{sig}**")
            except Exception:
                pass
        return "\n".join(lines)

    def _handle_target_correlation(self):
        """Show correlations of all numeric features with the target column."""
        target = self._get_target_col()
        if target is None:
            return "⚠️ No target column detected. Please select one from the sidebar."
        df = self.df
        if not pd.api.types.is_numeric_dtype(df[target]):
            return f"⚠️ Target `{target}` is not numeric. Cannot compute Pearson correlation directly."
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        if not num_cols:
            return "No numeric features to correlate with the target."

        corrs = []
        for c in num_cols:
            r = df[[c, target]].dropna().corr().iloc[0, 1]
            corrs.append((c, round(r, 4)))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)

        lines = [f"**Correlation of Features with `{target}`:**\n"]
        for feat, r in corrs:
            if abs(r) > 0.5:
                emoji = "🔴 Strong"
            elif abs(r) > 0.3:
                emoji = "🟡 Moderate"
            elif abs(r) > 0.1:
                emoji = "🔵 Weak"
            else:
                emoji = "⚪ Negligible"
            direction = "positive" if r > 0 else "negative"
            lines.append(f"- `{feat}`: **{r}** ({emoji} {direction})")
        return "\n".join(lines)

    # ─── ADVISORY / RECOMMENDATION QUESTIONS ─────────────────────
    def _is_advisory_question(self, q):
        """Detect advisory/recommendation questions that need domain-knowledge answers."""
        advisory_patterns = [
            'should i', 'how can i', 'how to', 'what steps', 'suggest',
            'recommend', 'advice', 'preprocessing', 'preprocess',
            'scaling', 'encoding', 'encode', 'normalize', 'normalization',
            'feature engineering', 'handle class imbalance', 'handle imbalance',
            'what model', 'which model', 'best model', 'work best',
            'remove to improve', 'features should be removed',
            'improve model', 'why are some', 'why do some', 'more likely',
            'before training', 'apply before', 'needed for',
        ]
        return any(p in q for p in advisory_patterns)

    def _handle_advisory_question(self, q):
        """Generate data-driven advisory responses for common ML/EDA questions."""
        eda = self.full_eda_cache
        df = self.df
        target = self._get_target_col()

        # ── Preprocessing steps
        if any(w in q for w in ['preprocessing', 'preprocess', 'before training',
                                'what steps', 'apply before']):
            return self._advise_preprocessing(eda, target)

        # ── Scaling / encoding
        if any(w in q for w in ['scaling', 'encoding', 'encode', 'normalize', 'normalization',
                                'should i apply']):
            return self._advise_scaling_encoding(eda)

        # ── Feature engineering
        if 'feature engineering' in q or ('feature' in q and 'engineer' in q) or \
           ('feature' in q and 'suggest' in q) or ('feature' in q and 'idea' in q):
            return self._advise_feature_engineering(eda)

        # ── Feature removal
        if any(w in q for w in ['remove', 'drop', 'eliminate', 'should be removed']):
            return self._advise_feature_removal(eda, target)

        # ── Class imbalance handling
        if 'imbalance' in q or 'imbalanced' in q or 'class imbalance' in q:
            return self._advise_class_imbalance(target)

        # ── Best model / which model
        if any(w in q for w in ['what model', 'which model', 'best model', 'work best',
                                'what ml']):
            return self._advise_best_model(eda, target)

        # ── Why default / likely
        if any(w in q for w in ['why are some', 'why do some', 'more likely', 'likely to default']):
            return self._advise_why_default(target)

        # Generic fallback for advisory
        return self._advise_preprocessing(eda, target)

    def _advise_preprocessing(self, eda, target):
        lines = ["**🔧 Recommended Preprocessing Steps:**\n"]
        # 1. Missing values
        if eda['missing']:
            n_missing = len(eda['missing'])
            lines.append(f"**1. Handle Missing Values** ({n_missing} columns affected):")
            for c, info in sorted(eda['missing'].items(), key=lambda x: x[1]['pct'], reverse=True)[:5]:
                if info['pct'] > 40:
                    lines.append(f"   - `{c}` ({info['pct']}%) → Consider **dropping** this column")
                elif info['pct'] > 10:
                    lines.append(f"   - `{c}` ({info['pct']}%) → Use **median/mode imputation** or KNN imputer")
                else:
                    lines.append(f"   - `{c}` ({info['pct']}%) → **Simple imputation** (median/mode)")
        else:
            lines.append("**1. Missing Values:** ✅ None detected")

        # 2. Encoding
        cat_cols = list(eda.get('categorical_stats', {}).keys())
        if cat_cols:
            lines.append(f"\n**2. Encode Categorical Variables** ({len(cat_cols)} columns):")
            for c in cat_cols[:5]:
                nuniq = eda['categorical_stats'][c]['unique']
                if nuniq <= 5:
                    lines.append(f"   - `{c}` ({nuniq} unique) → **One-Hot Encoding**")
                elif nuniq <= 15:
                    lines.append(f"   - `{c}` ({nuniq} unique) → **Label/Ordinal Encoding**")
                else:
                    lines.append(f"   - `{c}` ({nuniq} unique) → **Target Encoding** or drop")

        # 3. Outliers
        if eda['outliers']:
            lines.append(f"\n**3. Handle Outliers** ({len(eda['outliers'])} columns):")
            for c, info in sorted(eda['outliers'].items(), key=lambda x: x[1]['pct'], reverse=True)[:3]:
                lines.append(f"   - `{c}`: {info['count']} outliers ({info['pct']}%) → **Cap/clip** or log-transform")

        # 4. Scaling
        lines.append("\n**4. Feature Scaling:** Apply StandardScaler or MinMaxScaler to numeric features")

        # 5. Imbalance
        if target and target in self.df.columns:
            vc = self.df[target].value_counts()
            if len(vc) >= 2:
                ratio = vc.iloc[0] / vc.iloc[-1] if vc.iloc[-1] > 0 else 0
                if ratio > 2:
                    lines.append(f"\n**5. Handle Imbalance:** Target `{target}` has {ratio:.1f}:1 ratio → Use SMOTE or class_weight='balanced'")

        return "\n".join(lines)

    def _advise_scaling_encoding(self, eda):
        lines = ["**📐 Scaling & Encoding Recommendations:**\n"]
        # Encoding
        cat_stats = eda.get('categorical_stats', {})
        if cat_stats:
            lines.append("**Categorical Encoding:**")
            for c, info in cat_stats.items():
                n = info['unique']
                if n == 2:
                    lines.append(f"- `{c}` (2 values) → **Binary Encoding** (0/1)")
                elif n <= 5:
                    lines.append(f"- `{c}` ({n} values) → **One-Hot Encoding**")
                elif n <= 15:
                    lines.append(f"- `{c}` ({n} values) → **Label Encoding** or **Target Encoding**")
                else:
                    lines.append(f"- `{c}` ({n} values) → **Target Encoding** or consider dropping")

        # Scaling
        num_stats = eda.get('numeric_stats', {})
        if num_stats:
            lines.append("\n**Numeric Scaling:**")
            for c, s in num_stats.items():
                skew = abs(s.get('skew', 0))
                if skew > 2:
                    lines.append(f"- `{c}` (skew={s['skew']}) → **Log/Box-Cox transform** first, then scale")
                else:
                    lines.append(f"- `{c}` → **StandardScaler** (mean=0, std=1)")

        lines.append("\n💡 **Tip:** Tree-based models (RF, XGBoost) don't require scaling. "
                     "Linear models (Logistic Reg, SVM) need scaling.")
        return "\n".join(lines)

    def _advise_feature_engineering(self, eda):
        lines = ["**🛠️ Feature Engineering Suggestions:**\n"]
        df = self.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = list(eda.get('categorical_stats', {}).keys())

        # Ratio features
        if len(num_cols) >= 2:
            lines.append("**1. Ratio Features:**")
            if 'loan_amount' in num_cols and 'income' in df.columns:
                lines.append("   - `loan_to_income = loan_amount / income` (debt burden)")
            if 'loan_amount' in num_cols and 'property_value' in df.columns:
                lines.append("   - `loan_to_value = loan_amount / property_value` (LTV ratio)")
            if 'Upfront_charges' in num_cols and 'loan_amount' in num_cols:
                lines.append("   - `charge_ratio = Upfront_charges / loan_amount`")

        # Binning
        lines.append("\n**2. Binning Continuous Variables:**")
        for c in num_cols[:3]:
            s = eda['numeric_stats'].get(c, {})
            if abs(s.get('skew', 0)) > 1:
                lines.append(f"   - `{c}` (skewed) → Create quantile-based bins")

        # Interaction features
        if cat_cols and num_cols:
            lines.append("\n**3. Interaction Features:**")
            lines.append(f"   - Combine `{cat_cols[0]}` with `{num_cols[0]}` for group-level stats")

        # Log transform
        skewed = [c for c in num_cols if abs(eda['numeric_stats'].get(c, {}).get('skew', 0)) > 2]
        if skewed:
            lines.append(f"\n**4. Log Transform** (highly skewed): {', '.join(f'`{c}`' for c in skewed[:5])}")

        # Missing indicator
        missing_cols = list(eda.get('missing', {}).keys())
        if missing_cols:
            lines.append(f"\n**5. Missing Indicator Columns:** Create binary flags for {', '.join(f'`{c}`' for c in missing_cols[:5])}")

        return "\n".join(lines)

    def _advise_feature_removal(self, eda, target):
        lines = ["**🗑️ Features to Consider Removing:**\n"]
        df = self.df
        removals = []

        # High missing
        for c, info in eda.get('missing', {}).items():
            if info['pct'] > 40:
                removals.append((c, f"High missing values ({info['pct']}%)"))

        # ID-like columns
        for c in df.columns:
            if c.lower() in ['id', 'index', 'unnamed: 0'] or (df[c].nunique() == len(df)):
                removals.append((c, "ID/index column (no predictive value)"))

        # Constant or near-constant
        for c in df.columns:
            if df[c].nunique() <= 1:
                removals.append((c, "Constant value — no variance"))
            elif c.lower() == 'year' and df[c].nunique() == 1:
                removals.append((c, "Single year — no variation"))

        # High cardinality categoricals
        for c, info in eda.get('categorical_stats', {}).items():
            if info['unique'] > 50:
                removals.append((c, f"High cardinality ({info['unique']} unique values)"))

        # High correlation duplicates
        for pair in eda.get('high_correlations', []):
            if abs(pair['corr']) > 0.9:
                removals.append((pair['col2'], f"Very high correlation ({pair['corr']}) with `{pair['col1']}`"))

        if removals:
            for c, reason in removals:
                lines.append(f"- **`{c}`**: {reason}")
        else:
            lines.append("✅ No obvious columns to remove. All seem potentially useful.")

        if self.training_report and self.training_report.get('feature_importance'):
            fi = self.training_report['feature_importance']
            low_imp = [(f, imp) for f, imp in fi.items() if imp < 0.01]
            if low_imp:
                lines.append("\n**Low Importance Features (from trained model):**")
                for f, imp in low_imp[:5]:
                    lines.append(f"- `{f}`: importance = {imp:.4f}")
        return "\n".join(lines)

    def _advise_class_imbalance(self, target):
        if target is None:
            return "⚠️ No target column detected. Please select one from the sidebar."
        df = self.df
        vc = df[target].value_counts()
        total = len(df)
        lines = [f"**⚖️ Class Imbalance Analysis for `{target}`:**\n"]
        for val, count in vc.items():
            lines.append(f"- **{val}**: {count:,} ({count/total*100:.2f}%)")

        if len(vc) >= 2:
            ratio = vc.iloc[0] / vc.iloc[-1] if vc.iloc[-1] > 0 else float('inf')
            lines.append(f"\n**Imbalance Ratio:** {ratio:.2f}:1\n")
            lines.append("**Recommended Techniques:**")
            if ratio > 5:
                lines.append("1. **SMOTE** (Synthetic Minority Oversampling) — best for severe imbalance")
                lines.append("2. **class_weight='balanced'** in model hyperparameters")
                lines.append("3. **Random Undersampling** of majority class")
                lines.append("4. **Ensemble methods** like BalancedRandomForest")
                lines.append("5. Use **F1-score / AUC-ROC** instead of accuracy for evaluation")
            elif ratio > 2:
                lines.append("1. **class_weight='balanced'** in model hyperparameters")
                lines.append("2. **SMOTE** or **ADASYN** oversampling")
                lines.append("3. **Stratified cross-validation** to maintain class ratios")
                lines.append("4. Focus on **precision, recall, and F1** metrics")
            else:
                lines.append("✅ Imbalance is mild. Standard training should work well.")
                lines.append("💡 Still use **stratified splits** for safety.")
        return "\n".join(lines)

    def _advise_best_model(self, eda, target):
        df = self.df
        rows = eda['overview']['rows']
        num_cols = len(eda.get('numeric_stats', {}))
        cat_cols = len(eda.get('categorical_stats', {}))

        task = self.task_type or ('classification' if target and df[target].nunique() <= 15 else 'regression')

        lines = [f"**🏆 Model Recommendations for `{target}` ({task}):**\n"]
        lines.append(f"Dataset: {rows:,} rows, {num_cols} numeric + {cat_cols} categorical features\n")

        if task == 'classification':
            lines.append("**Recommended Models (in order):**")
            lines.append("1. **Gradient Boosting (XGBoost/LightGBM)** — Best for tabular data with mixed features")
            lines.append("2. **Random Forest** — Robust, handles outliers well, less prone to overfitting")
            lines.append("3. **Logistic Regression** — Fast, interpretable baseline")
            if rows > 50000:
                lines.append("4. **LightGBM** — Faster than XGBoost for large datasets")
        else:
            lines.append("**Recommended Models (in order):**")
            lines.append("1. **Gradient Boosting Regressor** — Best for complex non-linear relationships")
            lines.append("2. **Random Forest Regressor** — Handles outliers and missing patterns")
            lines.append("3. **Ridge/Lasso Regression** — Good if features are linearly related")

        # Specific recommendations based on data characteristics
        lines.append("\n**Data-Specific Notes:**")
        if eda.get('outliers') and len(eda['outliers']) > 3:
            lines.append("- Many outliers → Tree-based models (RF/GB) are more robust than linear models")
        if cat_cols > 5:
            lines.append("- Many categorical features → LightGBM handles categoricals natively")
        skewed = sum(1 for s in eda.get('numeric_stats', {}).values() if abs(s.get('skew', 0)) > 2)
        if skewed > 2:
            lines.append(f"- {skewed} highly skewed features → Apply log-transform before linear models")

        if self.training_report:
            r = self.training_report
            lines.append(f"\n**Current Results:** Best = **{r['best_model']}** (score: {r['best_score']:.4f})")
        return "\n".join(lines)

    def _advise_why_default(self, target):
        """Explain what factors drive the target variable using data analysis."""
        if target is None:
            return "⚠️ No target column detected. Please select one from the sidebar."
        df = self.df
        lines = [f"**🔍 Key Factors Influencing `{target}`:**\n"]

        # Feature importance if available
        if self.training_report and self.training_report.get('feature_importance'):
            fi = self.training_report['feature_importance']
            top = list(fi.items())[:8]
            lines.append("**Top Predictive Features (from trained model):**")
            for feat, imp in top:
                bar = "█" * int(imp * 40)
                lines.append(f"- `{feat}`: {imp:.4f} {bar}")

        # Group comparisons for top numeric features
        if pd.api.types.is_numeric_dtype(df[target]) or df[target].nunique() <= 10:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in num_cols:
                num_cols.remove(target)
            lines.append("\n**Mean Comparison by Target Group:**")
            for c in num_cols[:6]:
                try:
                    grp = df.groupby(target)[c].mean()
                    vals = ", ".join(f"{target}={k}: {v:.2f}" for k, v in grp.items())
                    lines.append(f"- `{c}`: {vals}")
                except Exception:
                    pass
        return "\n".join(lines)

    def _handle_column_query(self, q, eda):
        """Handle column-specific queries like 'average of X', 'max income'."""
        df = self.df
        # Find which column the user is asking about
        matched_col = None
        for c in df.columns:
            if c.lower() in q:
                matched_col = c
                break
        if matched_col is None:
            for c in df.columns:
                for word in c.lower().replace('_', ' ').split():
                    if len(word) > 2 and word in q:
                        matched_col = c
                        break
                if matched_col:
                    break

        if matched_col and pd.api.types.is_numeric_dtype(df[matched_col]):
            s = eda['numeric_stats'].get(matched_col, {})
            if any(w in q for w in ['average', 'mean', 'avg']):
                val = s.get('mean', round(df[matched_col].mean(), 4))
                return f"**Average of `{matched_col}`:** {val}"
            if any(w in q for w in ['maximum', 'max', 'highest', 'largest', 'top']):
                val = s.get('max', df[matched_col].max())
                return f"**Maximum of `{matched_col}`:** {val}"
            if any(w in q for w in ['minimum', 'min', 'lowest', 'smallest']):
                val = s.get('min', df[matched_col].min())
                return f"**Minimum of `{matched_col}`:** {val}"
            if any(w in q for w in ['median', 'middle']):
                val = s.get('median', round(df[matched_col].median(), 4))
                return f"**Median of `{matched_col}`:** {val}"
            if any(w in q for w in ['std', 'standard deviation', 'deviation']):
                val = s.get('std', round(df[matched_col].std(), 4))
                return f"**Std Dev of `{matched_col}`:** {val}"
            # Default: show full stats for that column
            return (f"**Stats for `{matched_col}`:**\n"
                    f"Mean={s.get('mean')}, Median={s.get('median')}, Std={s.get('std')}, "
                    f"Min={s.get('min')}, Max={s.get('max')}, Skew={s.get('skew')}, "
                    f"Q1={s.get('q25')}, Q3={s.get('q75')}")

        if matched_col and matched_col in eda.get('categorical_stats', {}):
            info = eda['categorical_stats'][matched_col]
            top = list(info['top_values'].items())[:5]
            top_str = "\n".join(f"- {k}: {v}" for k, v in top)
            return f"**`{matched_col}`** ({info['unique']} unique values)\n\n**Top values:**\n{top_str}"

        # Fallback
        return (f"I analyzed your question but couldn't find a specific match.\n\n"
                f"**Available columns:** {', '.join(df.columns.tolist())}\n\n"
                f"Try asking about: statistics, missing values, correlations, outliers, "
                f"distribution, predictions, probability, or specific column stats.")
