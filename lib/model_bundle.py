#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Portable interface for hierarchical classifier PKL bundles.

Place this module in ``../lib`` next to ``NeuralNetwork.py`` and
``nn__algorithm_selector.py``.  ``training/run.py`` serializes a ModelBundle
instance, and the deployed classifier package receives a copy of this module so
the PKL can be loaded there as well.
"""

from __future__ import annotations

import copy
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


BUNDLE_INTERFACE_VERSION = 2


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


class ModelBundle:
    """Data + interface stored in ``model.pkl``.

    The object deliberately exposes both dictionary-like access (for simple
    scripts) and explicit methods (for stable inter-program communication).
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = dict(data)
        self.data.setdefault("interface_version", BUNDLE_INTERFACE_VERSION)
        self.data.setdefault("created_at", datetime.now().astimezone().isoformat(timespec="seconds"))

    # ------------------------------------------------------------------
    # Minimal mapping compatibility
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    # ------------------------------------------------------------------
    # Requested public interface
    # ------------------------------------------------------------------
    def get_creation_date(self) -> str:
        """Return ISO-formatted creation date/time."""
        return str(self.data.get("created_at", ""))

    def get_project_name(self) -> str:
        """Return the training project name."""
        return str(self.data.get("project", ""))

    def get_author(self) -> str:
        """Return model author metadata."""
        return str(self.data.get("author", ""))

    def get_description(self) -> str:
        """Return model description metadata."""
        return str(self.data.get("description", ""))

    def get_version(self) -> str:
        """Return user-supplied model/data version metadata."""
        return str(self.data.get("version", ""))

    def get_contact_details(self) -> str:
        """Return author/contact metadata."""
        return str(self.data.get("contact_details", ""))

    def get_metadata(self) -> Dict[str, str]:
        """Return the human-facing model metadata as one dictionary."""
        return {
            "creation_date": self.get_creation_date(),
            "project": self.get_project_name(),
            "author": self.get_author(),
            "description": self.get_description(),
            "version": self.get_version(),
            "contact_details": self.get_contact_details(),
        }

    def get_training_genomes(self) -> List[str]:
        """Return all genome identifiers used for training."""
        return [str(v) for v in self.data.get("training", {}).get("genome_names", [])]

    def get_clusters(self, delimiter: str = "|") -> List[str]:
        """Return all unique hierarchical cluster paths.

        Both internal and terminal cluster paths are returned.  The root is not
        included.  Cluster components are re-joined with the caller-supplied
        delimiter.
        """
        if delimiter == "":
            raise ValueError("delimiter cannot be empty")

        paths: List[str] = []
        for labels in self.data.get("label_paths", []):
            labels = [str(v) for v in labels if str(v) != ""]
            for depth in range(1, len(labels) + 1):
                paths.append(delimiter.join(labels[:depth]))
        return _unique_preserve_order(paths)

    def get_cluster_genomes(self, delimiter: str = "|") -> List[str]:
        """Return ``label_A|label_B|...|genome_name`` strings."""
        if delimiter == "":
            raise ValueError("delimiter cannot be empty")

        genomes = self.get_training_genomes()
        label_paths = self.data.get("label_paths", [])
        if len(genomes) != len(label_paths):
            raise ValueError(
                "Bundle metadata is inconsistent: genome_names and label_paths "
                "have different lengths."
            )

        result: List[str] = []
        for labels, genome in zip(label_paths, genomes):
            parts = [str(v) for v in labels if str(v) != ""]
            parts.append(str(genome))
            result.append(delimiter.join(parts))
        return result

    def get_training_matrix(self) -> List[List[str]]:
        """Return a deep copy of the initial training matrix."""
        return copy.deepcopy(self.data.get("training_matrix", []))

    def get_model_tree(self) -> Dict[str, Any]:
        """Return a deep copy of the hierarchical model/profile tree."""
        return copy.deepcopy(self.data.get("tree", {}))

    def get_cluster_profile(
        self,
        cluster_path: str | Sequence[str] | None = None,
        delimiter: str = "|",
    ) -> Dict[str, Any]:
        """Return the stored allele-profile matrix for one cluster node.

        ``cluster_path=None`` or an empty path returns the root profile.
        """
        tree = self.data.get("tree") or {}
        if cluster_path is None or cluster_path == "":
            node = tree
        else:
            path = self._normalize_path(cluster_path, delimiter)
            node = self._find_node(tree, path)
        return copy.deepcopy(node.get("similarity_profile", {}))

    def get_hierarchy(self, format: str = "text", delimiter: str = "|") -> str:
        """Return hierarchy as ``text`` or ``newick``.

        Splitting nodes are annotated with the neural-network / classifier
        algorithm stored at that node.
        """
        fmt = str(format).strip().lower()
        if fmt in {"text", "txt", "tab", "indented"}:
            return self._hierarchy_text(delimiter=delimiter)
        if fmt in {"newick", "nwk"}:
            return self._hierarchy_newick()
        raise ValueError("format must be 'text' or 'newick'")

    def get_terminal_cluster_count(self) -> int:
        """Return number of distinct terminal cluster paths in training data."""
        paths = {
            tuple(str(v) for v in labels if str(v) != "")
            for labels in self.data.get("label_paths", [])
        }
        paths.discard(tuple())
        return len(paths)

    def get_interface_info(self) -> Dict[str, Any]:
        """Return a compact description of the bundle API/data."""
        return {
            "interface_version": self.data.get("interface_version", BUNDLE_INTERFACE_VERSION),
            "created_at": self.get_creation_date(),
            "project": self.get_project_name(),
            "author": self.get_author(),
            "description": self.get_description(),
            "version": self.get_version(),
            "contact_details": self.get_contact_details(),
            "training_genomes": len(self.get_training_genomes()),
            "terminal_clusters": self.get_terminal_cluster_count(),
            "hierarchy_levels": int(self.data.get("hierarchy_levels", 0)),
            "algorithm_mode": self.data.get("algorithm_mode"),
            "min_cluster_size": self.data.get("min_cluster_size"),
            "max_entropy": self.data.get("max_entropy"),
        }
        
    def print_info(self):
        fileds = [
            "interface_version",
            "created_at",
            "project",
            "author",
            "description",
            "version",
            "contact_details",
            "training_genomes",
            "terminal_clusters",
            "hierarchy_levels",
            "algorithm_mode",
            "min_cluster_size",
            "max_entropy"
        ]
        info = self.get_interface_info()
        print()
        for key in fields:
            print(info[key])
        print()


    def show_cluster_graph(
        self,
        output_folder: str | Path,
        delimiter: str = "|",
    ) -> str:
        """Display the stored hierarchical classifier as a Matplotlib tree.

        Internal/splitting nodes are drawn as small squares coloured by the
        classifier algorithm stored at the node. Nodes without a trained
        classifier are black.

        Terminal nodes are drawn as coloured circles. Their colour represents
        the final terminal category (the last hierarchy component). Terminal
        nodes remain numbered on the graph; the numbered mappings of their last
        two hierarchy levels are written to ``graph_legend.txt``.

        Training accuracy is printed beside a splitting node when a finite
        ``training_accuracy`` value is stored in its model. Missing accuracy
        values are not displayed.

        The graph legend contains both classifier-colour squares and
        terminal-category-colour circles.

        Returns
        -------
        str
            Path to the saved ``graph_legend.txt`` file.
        """
        if not delimiter:
            raise ValueError("delimiter cannot be empty")

        try:
            import math
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError as exc:
            raise RuntimeError(
                "Matplotlib is required to display the cluster graph. "
                "Install it with 'python -m pip install matplotlib'."
            ) from exc

        tree = self.data.get("tree") or {}
        if not tree:
            raise ValueError("The model bundle contains no cluster tree.")

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        legend_path = output_path / "graph_legend.txt"

        # ------------------------------------------------------------------
        # Flatten the hierarchy while retaining parent relationships.
        # ------------------------------------------------------------------
        nodes = []

        def walk(node, path, parent_index, depth):
            index = len(nodes)

            nodes.append({
                "node": node,
                "path": tuple(path),
                "parent": parent_index,
                "depth": depth,
                "children": [],
            })

            if parent_index is not None:
                nodes[parent_index]["children"].append(index)

            for child_label, child_node in node.get("children", {}).items():
                walk(
                    child_node,
                    path + (str(child_label),),
                    index,
                    depth + 1,
                )

            return index

        walk(tree, tuple(), None, 0)

        terminal_indices = [
            index
            for index, item in enumerate(nodes)
            if not item["children"]
        ]

        terminal_number = {
            index: number
            for number, index in enumerate(terminal_indices, start=1)
        }

        # ------------------------------------------------------------------
        # Stable tree layout.
        # ------------------------------------------------------------------
        positions = {}

        for x, index in enumerate(terminal_indices):
            positions[index] = (
                float(x),
                -float(nodes[index]["depth"]),
            )

        for index in sorted(
            range(len(nodes)),
            key=lambda i: nodes[i]["depth"],
            reverse=True,
        ):
            if index in positions:
                continue

            children = nodes[index]["children"]

            if children:
                x = (
                    sum(positions[child][0] for child in children)
                    / len(children)
                )
            else:
                x = 0.0

            positions[index] = (
                x,
                -float(nodes[index]["depth"]),
            )

        # ------------------------------------------------------------------
        # Classifier colours for internal/splitting nodes.
        # ------------------------------------------------------------------
        algorithms = sorted(
            {
                self._node_algorithm(item["node"])
                for item in nodes
                if (
                    item["children"]
                    and self._node_algorithm(item["node"])
                )
            },
            key=str.casefold,
        )

        algorithm_cmap = plt.get_cmap("tab20")

        algorithm_colors = {
            name: algorithm_cmap(index % algorithm_cmap.N)
            for index, name in enumerate(algorithms)
        }

        # ------------------------------------------------------------------
        # Terminal-category colours.
        # The final hierarchy component determines the category colour.
        # ------------------------------------------------------------------
        terminal_categories = sorted(
            {
                str(nodes[index]["path"][-1])
                for index in terminal_indices
                if nodes[index]["path"]
            },
            key=str.casefold,
        )

        # tab20 is easy to distinguish for modest category numbers. For larger
        # sets use a resampled HSV map so every terminal category still gets a
        # separate colour.
        if len(terminal_categories) <= 20:
            terminal_cmap = plt.get_cmap("tab20")
            terminal_colors = {
                category: terminal_cmap(index % terminal_cmap.N)
                for index, category in enumerate(terminal_categories)
            }
        else:
            terminal_cmap = plt.get_cmap(
                "hsv",
                max(1, len(terminal_categories)),
            )
            terminal_colors = {
                category: terminal_cmap(index)
                for index, category in enumerate(terminal_categories)
            }

        # ------------------------------------------------------------------
        # Stored node accuracy.
        # ------------------------------------------------------------------
        def accuracy(node):
            model = node.get("model")

            candidates = []

            if isinstance(model, dict):
                candidates.extend([
                    model.get("training_accuracy"),
                    model.get("accuracy"),
                    model.get("cv_accuracy"),
                    model.get("validation_accuracy"),
                    model.get("score"),
                    model.get("best_score"),
                ])

                metrics = model.get("metrics")

                if isinstance(metrics, dict):
                    candidates.extend([
                        metrics.get("accuracy"),
                        metrics.get("score"),
                    ])

            candidates.extend([
                node.get("training_accuracy"),
                node.get("accuracy"),
                node.get("cv_accuracy"),
                node.get("score"),
            ])

            for value in candidates:
                if value is None:
                    continue

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                if math.isfinite(value):
                    return value

            return None

        # ------------------------------------------------------------------
        # Draw graph.
        # ------------------------------------------------------------------
        n_leaves = max(1, len(terminal_indices))
        max_depth = max(item["depth"] for item in nodes)

        fig_width = max(
            10.0,
            min(32.0, 0.32 * n_leaves + 8.0),
        )

        fig_height = max(
            6.0,
            min(24.0, 1.25 * (max_depth + 1) + 3.0),
        )

        fig, ax = plt.subplots(
            figsize=(fig_width, fig_height)
        )

        # Draw orthogonal branches first.
        for index, item in enumerate(nodes):
            parent = item["parent"]

            if parent is None:
                continue

            x1, y1 = positions[parent]
            x2, y2 = positions[index]

            ax.plot(
                [x1, x1, x2],
                [y1, y2, y2],
                linewidth=0.8,
                color="0.55",
                zorder=1,
            )

        # Draw nodes.
        for index, item in enumerate(nodes):
            x, y = positions[index]

            if index in terminal_number:
                # Terminal node: coloured circle by final ecological category.
                terminal_category = (
                    str(item["path"][-1])
                    if item["path"]
                    else ""
                )

                color = terminal_colors.get(
                    terminal_category,
                    "black",
                )

                ax.scatter(
                    [x],
                    [y],
                    marker="o",
                    s=42,
                    color=color,
                    edgecolors="black",
                    linewidths=0.35,
                    zorder=3,
                )

                ax.text(
                    x,
                    y - 0.18,
                    str(terminal_number[index]),
                    fontsize=7,
                    va="top",
                    ha="center",
                )

                # Terminal nodes have no splitting classifier, therefore no
                # training accuracy is shown.

            else:
                # Internal/splitting node: square coloured by classifier.
                algorithm = self._node_algorithm(
                    item["node"]
                )

                color = (
                    algorithm_colors.get(
                        algorithm,
                        "black",
                    )
                    if algorithm
                    else "black"
                )

                ax.scatter(
                    [x],
                    [y],
                    marker="s",
                    s=38,
                    color=color,
                    zorder=3,
                )

                acc = accuracy(item["node"])

                if acc is not None:
                    ax.text(
                        x + 0.08,
                        y + 0.08,
                        f"{acc:.2f}",
                        fontsize=7,
                        va="bottom",
                        ha="left",
                    )

                if index == 0:
                    ax.text(
                        x,
                        y + 0.20,
                        "Root",
                        fontsize=8,
                        va="bottom",
                        ha="center",
                    )

        # ------------------------------------------------------------------
        # Combined legend:
        #   squares = splitting-node classifiers
        #   circles = terminal ecological categories
        # ------------------------------------------------------------------
        legend_handles = []

        # Section heading for classifiers.
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=None,
                label="Node classifiers",
            )
        )

        legend_handles.extend([
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markersize=7,
                markerfacecolor=algorithm_colors[name],
                markeredgecolor=algorithm_colors[name],
                label=name,
            )
            for name in algorithms
        ])

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markersize=7,
                markerfacecolor="black",
                markeredgecolor="black",
                label="No NN model",
            )
        )

        # Blank separator between the two legend sections.
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=None,
                label=" ",
            )
        )

        # Section heading for terminal categories.
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=None,
                label="Terminal categories",
            )
        )

        legend_handles.extend([
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=terminal_colors[category],
                markeredgecolor="black",
                markeredgewidth=0.35,
                label=category,
            )
            for category in terminal_categories
        ])

        ax.legend(
            handles=legend_handles,
            title="Model graph legend",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            ncol=1,
        )

        ax.set_title(
            f"{self.get_project_name()} hierarchical classifier"
        )

        ax.set_axis_off()
        ax.margins(x=0.03, y=0.10)

        # Leave room on the right for the combined legend.
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

        # ------------------------------------------------------------------
        # Save numbered terminal-node text legend.
        # ------------------------------------------------------------------
        legend_lines = []

        for index in terminal_indices:
            path = nodes[index]["path"]
            visible = (
                path[-2:]
                if len(path) >= 2
                else path
            )

            legend_lines.append(
                f"{terminal_number[index]}\t"
                f"{delimiter.join(visible)}"
            )

        legend_path.write_text(
            "\n".join(legend_lines) + "\n",
            encoding="utf-8-sig",
        )

        print(
            f"Graph legend saved to: {legend_path}"
        )

        # Show an interactive Matplotlib window. The toolbar can be used to
        # save the graph in PNG, PDF, SVG and other supported formats.
        plt.show()
        plt.close(fig)

        return str(legend_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, file_path: str | Path) -> str:
        """Save this bundle to a PKL file and return its path."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return str(path)

    @classmethod
    def load(cls, file_path: str | Path) -> "ModelBundle":
        """Load a ModelBundle from a PKL file."""
        path = Path(file_path)
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            # Backward compatibility with early dictionary bundles.
            return cls(obj)
        raise TypeError(f"{path} does not contain a compatible ModelBundle")

    # ------------------------------------------------------------------
    # Hierarchy presentation
    # ------------------------------------------------------------------
    @staticmethod
    def _node_algorithm(node: Dict[str, Any]) -> str:
        model = node.get("model")
        if isinstance(model, dict):
            return str(model.get("algorithm", "") or "")
        return ""

    def _hierarchy_text(self, delimiter: str = "|") -> str:
        tree = self.data.get("tree") or {}
        lines: List[str] = []

        def node_suffix(node: Dict[str, Any]) -> str:
            algorithm = self._node_algorithm(node)
            if algorithm:
                return f"  [algorithm={algorithm}]"
            if node.get("children"):
                reason = str(node.get("model_skip_reason", "") or "")
                return f"  [algorithm=None; fallback=allele-profile{'; ' + reason if reason else ''}]"
            return ""

        def walk(node: Dict[str, Any], depth: int, label: str) -> None:
            lines.append("\t" * depth + f"{label}{node_suffix(node)}")
            for child_label, child_node in node.get("children", {}).items():
                walk(child_node, depth + 1, str(child_label))

        lines.append(f"<root>{node_suffix(tree)}")
        for child_label, child_node in tree.get("children", {}).items():
            walk(child_node, 1, str(child_label))
        return "\n".join(lines)

    @staticmethod
    def _newick_escape(label: str) -> str:
        text = str(label)
        if any(ch in text for ch in " ,:;()[]'"):
            return "'" + text.replace("'", "''") + "'"
        return text

    def _hierarchy_newick(self) -> str:
        tree = self.data.get("tree") or {}

        def render(node: Dict[str, Any], label: str) -> str:
            children = [
                render(child_node, str(child_label))
                for child_label, child_node in node.get("children", {}).items()
            ]
            algorithm = self._node_algorithm(node)
            node_label = str(label)
            if algorithm:
                node_label = f"{node_label}[{algorithm}]"
            elif children:
                node_label = f"{node_label}[None/profile]"
            escaped = self._newick_escape(node_label)
            if children:
                return "(" + ",".join(children) + ")" + escaped
            return escaped

        children = [
            render(child_node, str(child_label))
            for child_label, child_node in tree.get("children", {}).items()
        ]
        root_algorithm = self._node_algorithm(tree)
        if root_algorithm:
            root_label = f"<root>[{root_algorithm}]"
        elif children:
            root_label = "<root>[None/profile]"
        else:
            root_label = "<root>"
        if children:
            return "(" + ",".join(children) + ")" + self._newick_escape(root_label) + ";"
        return self._newick_escape(root_label) + ";"

    # ------------------------------------------------------------------
    # Grafting
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_path(path: str | Sequence[str], delimiter: str) -> Tuple[str, ...]:
        if isinstance(path, str):
            parts = [part.strip() for part in path.split(delimiter) if part.strip()]
        else:
            parts = [str(part).strip() for part in path if str(part).strip()]
        if not parts:
            raise ValueError("terminal_path cannot be empty")
        return tuple(parts)

    @staticmethod
    def _find_node(tree: Dict[str, Any], path: Sequence[str]) -> Dict[str, Any]:
        node = tree
        for part in path:
            children = node.get("children", {})
            if part not in children:
                raise KeyError(f"Cluster path does not exist: {'|'.join(path)}")
            node = children[part]
        return node

    @staticmethod
    def _prefix_tree_paths(node: Dict[str, Any], prefix: Tuple[str, ...]) -> Dict[str, Any]:
        node = copy.deepcopy(node)

        def walk(current: Dict[str, Any], current_path: Tuple[str, ...]) -> None:
            current["path"] = list(current_path)
            for child_label, child_node in current.get("children", {}).items():
                walk(child_node, current_path + (str(child_label),))

        walk(node, prefix)
        return node

    def _validate_graft_compatibility(self, child: "ModelBundle") -> None:
        if self.data.get("data_type") != child.data.get("data_type"):
            raise ValueError("Cannot graft bundles with different data types.")

        if list(self.data.get("feature_titles", [])) != list(child.data.get("feature_titles", [])):
            raise ValueError(
                "Cannot graft bundles with different feature-title sets/order. "
                "This restriction keeps allelic-state calling and stored pipelines coherent."
            )

        parent_ref = str(self.data.get("reference_name", "") or "")
        child_ref = str(child.data.get("reference_name", "") or "")
        if parent_ref and child_ref and parent_ref != child_ref:
            raise ValueError(
                f"Cannot graft bundles built against different references: "
                f"{parent_ref!r} != {child_ref!r}"
            )

    def graft(
        self,
        child_bundle: "ModelBundle" | str | Path,
        terminal_path: str | Sequence[str],
        *,
        delimiter: str = "|",
        new_project_name: str | None = None,
    ) -> "ModelBundle":
        """Refine a terminal cluster with the hierarchy from another bundle.

        The child bundle is expected to have been trained on the SAME genomes
        that currently occupy ``terminal_path`` in the parent bundle.  Their
        old terminal labels are replaced by ``terminal_path + child_labels``.

        This strict rule prevents duplicated genomes and prevents a cluster from
        becoming both terminal and internal after grafting.  Both source bundles
        remain unchanged; a NEW ModelBundle is returned.
        """
        child = (
            ModelBundle.load(child_bundle)
            if isinstance(child_bundle, (str, Path))
            else child_bundle
        )
        if not isinstance(child, ModelBundle):
            raise TypeError("child_bundle must be a ModelBundle or path to one")

        self._validate_graft_compatibility(child)
        target_path = self._normalize_path(terminal_path, delimiter)

        parent_genomes = self.get_training_genomes()
        parent_labels = [list(v) for v in self.data.get("label_paths", [])]
        if len(parent_genomes) != len(parent_labels):
            raise ValueError(
                "Parent bundle metadata is inconsistent: genome_names and "
                "label_paths have different lengths."
            )

        target_indices = [
            i
            for i, labels in enumerate(parent_labels)
            if tuple(str(v) for v in labels if str(v) != "") == target_path
        ]
        if not target_indices:
            raise ValueError(
                f"No training genomes terminate at cluster {delimiter.join(target_path)!r}."
            )

        target_genomes = [parent_genomes[i] for i in target_indices]
        child_genomes = child.get_training_genomes()
        child_labels = [list(v) for v in child.data.get("label_paths", [])]

        if len(child_genomes) != len(child_labels):
            raise ValueError(
                "Child bundle metadata is inconsistent: genome_names and "
                "label_paths have different lengths."
            )

        if set(target_genomes) != set(child_genomes):
            only_parent = sorted(set(target_genomes) - set(child_genomes))
            only_child = sorted(set(child_genomes) - set(target_genomes))
            raise ValueError(
                "The child bundle must contain exactly the genomes present in "
                f"terminal cluster {delimiter.join(target_path)!r}. "
                f"Missing from child: {only_parent[:10]}; "
                f"unexpected in child: {only_child[:10]}."
            )

        child_label_by_genome = {
            genome: [str(v) for v in labels if str(v) != ""]
            for genome, labels in zip(child_genomes, child_labels)
        }

        result = copy.deepcopy(self)
        target = self._find_node(result.data["tree"], target_path)

        if target.get("children"):
            raise ValueError(
                f"Target {delimiter.join(target_path)} is not terminal; "
                "it already has children."
            )

        child_root = copy.deepcopy(child.data.get("tree") or {})
        target["model"] = child_root.get("model")
        target["model_skip_reason"] = child_root.get("model_skip_reason", "")
        target["default_child"] = child_root.get("default_child")
        target["similarity_profile"] = copy.deepcopy(
            child_root.get("similarity_profile", target.get("similarity_profile", {}))
        )
        target["child_profiles"] = copy.deepcopy(child_root.get("child_profiles", {}))
        target["child_sample_counts"] = copy.deepcopy(
            child_root.get("child_sample_counts", {})
        )
        target["children"] = {}
        for child_label, child_node in child_root.get("children", {}).items():
            target["children"][child_label] = self._prefix_tree_paths(
                child_node,
                target_path + (str(child_label),),
            )

        # Replace the terminal membership paths of the target genomes.
        refined_paths = [list(v) for v in parent_labels]
        for i in target_indices:
            genome = parent_genomes[i]
            refined_paths[i] = list(target_path) + child_label_by_genome[genome]
        result.data["label_paths"] = refined_paths

        # Update the first-column hierarchy headings in the stored initial
        # training matrix, while preserving all feature cells as originally
        # supplied by the parent training matrix.
        combined_matrix = result.get_training_matrix()
        if combined_matrix and len(combined_matrix) == len(parent_genomes) + 1:
            stored_delimiter = str(result.data.get("delimiter", "|"))
            for i in target_indices:
                row_index = i + 1
                if row_index >= len(combined_matrix) or not combined_matrix[row_index]:
                    continue
                genome = parent_genomes[i]
                heading = refined_paths[i] + [genome]
                combined_matrix[row_index][0] = stored_delimiter.join(heading)
            result.data["training_matrix"] = combined_matrix

        result.data["hierarchy_levels"] = max(
            int(result.data.get("hierarchy_levels", 0)),
            len(target_path) + int(child.data.get("hierarchy_levels", 0)),
        )
        result.data["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if new_project_name is not None:
            result.data["project"] = str(new_project_name)

        result.data.setdefault("graft_history", []).append({
            "parent_project": self.get_project_name(),
            "child_project": child.get_project_name(),
            "terminal_path": delimiter.join(target_path),
            "refined_genomes": len(target_genomes),
            "created_at": result.data["created_at"],
        })
        return result

