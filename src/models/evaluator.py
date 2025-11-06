"""
Model evaluation utilities.

This module provides functions for comprehensive model evaluation
including metrics calculation, visualization, and reporting.
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate and compare machine learning models.
    """

    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            metrics["avg_precision"] = average_precision_score(y_test, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])

        # Classification report
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
            metrics["feature_importance"] = {
                k: float(v)
                for k, v in sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:20]
            }

        self.evaluation_results[model_name] = metrics

        logger.info(f"{model_name} Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        if "roc_auc" in metrics:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def compare_models(
        self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models on test data.

        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing models...")

        comparison_data = []

        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)

            comparison_data.append(
                {
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1"],
                    "ROC-AUC": metrics.get("roc_auc", np.nan),
                    "Avg Precision": metrics.get("avg_precision", np.nan),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("ROC-AUC", ascending=False)

        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        return comparison_df

    def plot_confusion_matrix(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        save_path: Optional[Path] = None,
    ):
        """
        Plot confusion matrix for a model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Outage", "Outage"],
            yticklabels=["No Outage", "Outage"],
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_roc_curve(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[Path] = None,
    ):
        """
        Plot ROC curves for multiple models.

        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))

        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curves saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_precision_recall_curve(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[Path] = None,
    ):
        """
        Plot precision-recall curves for multiple models.

        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))

        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                plt.plot(
                    recall,
                    precision,
                    label=f"{model_name} (AP = {avg_precision:.3f})",
                    linewidth=2,
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(
            "Precision-Recall Curves - Model Comparison", fontsize=14, fontweight="bold"
        )
        plt.legend(loc="best", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Precision-recall curves saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str,
        top_n: int = 20,
        save_path: Optional[Path] = None,
    ):
        """
        Plot feature importance for a model.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return

        # Get feature importance
        importance = model.feature_importances_
        feature_importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(feature_importance_df)),
            feature_importance_df["importance"],
            alpha=0.8,
        )
        plt.yticks(range(len(feature_importance_df)), feature_importance_df["feature"])
        plt.xlabel("Importance", fontsize=12)
        plt.title(
            f"Top {top_n} Feature Importance - {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def generate_report(
        self,
        output_path: Path,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Generate comprehensive evaluation report.

        Args:
            output_path: Path to save the report
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Evaluate all models
        comparison_df = self.compare_models(models, X_test, y_test)

        # Save comparison table
        comparison_path = output_path / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")

        # Plot ROC curves
        self.plot_roc_curve(models, X_test, y_test, output_path / "roc_curves.png")

        # Plot precision-recall curves
        self.plot_precision_recall_curve(
            models, X_test, y_test, output_path / "precision_recall_curves.png"
        )

        # Plot confusion matrices for each model
        for model_name, model in models.items():
            safe_name = model_name.replace(" ", "_").lower()
            self.plot_confusion_matrix(
                model,
                X_test,
                y_test,
                model_name,
                output_path / f"confusion_matrix_{safe_name}.png",
            )

            # Plot feature importance if available
            if hasattr(model, "feature_importances_"):
                self.plot_feature_importance(
                    model,
                    X_test.columns.tolist(),
                    model_name,
                    save_path=output_path / f"feature_importance_{safe_name}.png",
                )

        # Save detailed metrics
        metrics_path = output_path / "detailed_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        logger.info(f"Detailed metrics saved to {metrics_path}")

        # Create markdown report
        self._create_markdown_report(output_path, comparison_df)

    def _create_markdown_report(self, output_path: Path, comparison_df: pd.DataFrame):
        """
        Create a markdown summary report.

        Args:
            output_path: Directory to save the report
            comparison_df: DataFrame with model comparison
        """
        report_path = output_path / "evaluation_report.md"

        with open(report_path, "w") as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(
                f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## Model Comparison\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")

            f.write("## Best Model\n\n")
            best_model = comparison_df.iloc[0]
            f.write(f"**Model**: {best_model['Model']}\n\n")
            f.write(f"- **Accuracy**: {best_model['Accuracy']:.4f}\n")
            f.write(f"- **Precision**: {best_model['Precision']:.4f}\n")
            f.write(f"- **Recall**: {best_model['Recall']:.4f}\n")
            f.write(f"- **F1 Score**: {best_model['F1 Score']:.4f}\n")
            f.write(f"- **ROC-AUC**: {best_model['ROC-AUC']:.4f}\n\n")

            f.write("## Visualizations\n\n")
            f.write("### ROC Curves\n")
            f.write("![ROC Curves](roc_curves.png)\n\n")
            f.write("### Precision-Recall Curves\n")
            f.write("![Precision-Recall Curves](precision_recall_curves.png)\n\n")

            for model_name in comparison_df["Model"]:
                safe_name = model_name.replace(" ", "_").lower()
                f.write(f"### {model_name} - Confusion Matrix\n")
                f.write(
                    f"![Confusion Matrix](" f"confusion_matrix_{safe_name}.png)\n\n"
                )

        logger.info(f"Markdown report saved to {report_path}")
