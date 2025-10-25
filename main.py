"""
Dengue Transmission Network Analysis using Discrete Morse Theory (FAST)
======================================================================
This script identifies critical transmission hubs in Brazil's dengue network
by applying discrete Morse theory to municipality-level data from SINAN.

Changes vs original:
- Sparsify temporal edges: keep only TOPK strongest correlations per node.
- Raise correlation threshold (configurable).
- Correct weight semantics: use distances (smaller = stronger).
- Convert import-count edges to distances (1/(1+count)).
- Optionally restrict centrality to the largest SCC.
- Use unweighted betweenness or small-k weighted betweenness.
- Multiple speed knobs at the top.
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ------------------------- SPEED / QUALITY KNOBS -------------------------
# Temporal correlation graph sparsity
MIN_CORRELATION = 0.7     # raise to 0.75–0.85 to sparsify more
TOPK_TEMPORAL = 15        # top-K neighbors per node by correlation (10–25 recommended)

# Node filtering
MIN_CASES_PER_NODE = 0    # e.g., 25 to drop tiny/noisy municipalities early

# Betweenness centrality settings
USE_WEIGHTED_BETWEENNESS = False  # True if you want weighted betweenness (slower)
BETWEENNESS_K = 100               # sample size (50–200 typical). Ignored if USE_WEIGHTED_BETWEENNESS=False.
BETWEENNESS_SEED = 42             # reproducibility

# Component focus: compute centrality on the largest strongly connected component
FOCUS_ON_GIANT_SCC = True

# Date parsing (kept as in original)
DATE_FIELDS = ['dt_sin_pri', 'dt_notific', 'dt_invest']
DATE_FMT = '%Y-%m-%d'  # adjust if your CSV uses another format

# ------------------------------------------------------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dengue_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DengueNetworkAnalyzer:
    """
    Analyzes dengue transmission networks using Discrete Morse Theory
    to identify critical transmission hubs.
    """
    def __init__(self, csv_path):
        """Initialize the analyzer with the path to SINAN data."""
        self.csv_path = csv_path
        self.df = None
        self.graph = None
        self.municipality_scores = {}
        self.critical_points = {}

    def load_and_preprocess_data(self):
        """
        Load SINAN dengue data and perform initial preprocessing.

        Key fields (if available):
        - id_mn_resi: Municipality of residence
        - comuninf: Municipality of probable infection
        - tpautocto: Is case autochthonous? (1=Yes, 2=No, 3=Indeterminate)
        - classi_fin: Final classification (1=Confirmed, 2=Discarded)
        - dt_sin_pri: Date of first symptoms
        - sem_pri: Epidemiological week
        """
        logger.info("="*70)
        logger.info("STEP 1: LOADING AND PREPROCESSING DATA")
        logger.info("="*70)

        try:
            self.df = pd.read_csv(self.csv_path, encoding='latin-1', low_memory=False)
            logger.info(f"✓ Loaded {len(self.df):,} records from {self.csv_path}")
            logger.info(f"✓ Columns available: {len(self.df.columns)}")
            self.df.columns = self.df.columns.str.lower()
        except Exception as e:
            logger.error(f"✗ Failed to load data: {e}")
            raise

        # Filter to confirmed cases if available; otherwise keep all
        initial_count = len(self.df)
        if 'classi_fin' in self.df.columns:
            logger.info(f"\n📊 classi_fin values distribution:\n{self.df['classi_fin'].value_counts()}")
            confirmed = self.df[self.df['classi_fin'] == 1].copy()
            if len(confirmed) > 0:
                self.df = confirmed
                logger.info(f"✓ Filtered to confirmed cases: {len(self.df):,} ({len(self.df)/initial_count*100:.1f}%)")
            else:
                logger.warning("⚠️ No confirmed cases found; proceeding with ALL records.")
        else:
            logger.warning("⚠️ 'classi_fin' not found; using all records.")

        # Parse dates if present
        for field in DATE_FIELDS:
            if field in self.df.columns:
                logger.info(f"✓ Converting {field} to datetime")
                self.df[field] = pd.to_datetime(self.df[field], format=DATE_FMT, errors='coerce')

        # Municipality codes
        if 'id_mn_resi' not in self.df.columns:
            logger.error("✗ Column 'id_mn_resi' not found!")
            raise KeyError("Required column 'id_mn_resi' not found")

        logger.info(f"\n📊 Municipality data quality:")
        logger.info(f"   - Total records: {len(self.df):,}")
        logger.info(f"   - Non-null id_mn_resi: {self.df['id_mn_resi'].notna().sum():,}")
        logger.info(f"   - Null id_mn_resi: {self.df['id_mn_resi'].isna().sum():,}")

        self.df['id_mn_resi'] = self.df['id_mn_resi'].astype(str).str.strip()
        before_drop = len(self.df)
        self.df = self.df[self.df['id_mn_resi'] != 'nan']
        self.df = self.df.dropna(subset=['id_mn_resi'])
        dropped = before_drop - len(self.df)
        if dropped > 0:
            logger.info(f"✓ Removed {dropped:,} records with missing residence municipality")

        if len(self.df) == 0:
            logger.error("✗ ALL records removed after filtering! No valid municipality data.")
            raise ValueError("No valid data remaining after filtering")

        if 'comuninf' in self.df.columns:
            self.df['comuninf'] = self.df['comuninf'].astype(str).str.strip()
        else:
            logger.warning("⚠️ 'comuninf' (infection location) not found - limiting import-flow edges")

        # Optional: pre-filter nodes with few cases
        if MIN_CASES_PER_NODE > 0:
            counts = self.df['id_mn_resi'].value_counts()
            keep_munis = set(counts[counts >= MIN_CASES_PER_NODE].index)
            before = len(self.df)
            self.df = self.df[self.df['id_mn_resi'].isin(keep_munis)].copy()
            logger.info(f"✓ Prefiltered by case count ≥{MIN_CASES_PER_NODE}: {before:,} → {len(self.df):,} rows; {len(keep_munis):,} municipalities")

        logger.info(f"\n📊 Data Summary:")
        logger.info(f"   - Total cases: {len(self.df):,}")
        logger.info(f"   - Unique municipalities (residence): {self.df['id_mn_resi'].nunique():,}")

        if 'dt_sin_pri' in self.df.columns:
            valid_dates = self.df['dt_sin_pri'].notna().sum()
            logger.info(f"   - Records with valid dt_sin_pri: {valid_dates:,}")
            if valid_dates > 0:
                logger.info(f"   - Date range: {self.df['dt_sin_pri'].min()} to {self.df['dt_sin_pri'].max()}")

        if 'tpautocto' in self.df.columns:
            autoc_dist = self.df['tpautocto'].value_counts()
            logger.info(f"\n   Autochthony distribution:")
            logger.info(f"   - Autochthonous (local): {autoc_dist.get(1, 0):,}")
            logger.info(f"   - Imported: {autoc_dist.get(2, 0):,}")
            logger.info(f"   - Indeterminate: {autoc_dist.get(3, 0):,}")
        else:
            logger.warning("   ⚠️ 'tpautocto' not found - cannot distinguish imported vs local cases")

        return self.df

    def build_transmission_network(self):
        """
        Build a directed graph representing dengue transmission between municipalities.

        Edges:
        1) Import edges (comuninf -> id_mn_resi), weighted by distance = 1/(1+count)
        2) Temporal correlation edges: keep only TOPK strongest per node, weight = 1 - corr
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: BUILDING TRANSMISSION NETWORK (SPARSE & CORRECT WEIGHTS)")
        logger.info("="*70)

        self.graph = nx.DiGraph()

        # Add nodes
        municipalities = set(self.df['id_mn_resi'].unique())
        self.graph.add_nodes_from(municipalities)
        logger.info(f"✓ Added {len(municipalities):,} municipalities as nodes")

        # Import-flow edges (non-autochthonous)
        if 'tpautocto' in self.df.columns and 'comuninf' in self.df.columns:
            imported_cases = self.df[
                (self.df['tpautocto'] == 2) &
                (self.df['comuninf'].notna()) &
                (self.df['comuninf'] != 'nan')
            ].copy()

            logger.info(f"✓ Found {len(imported_cases):,} imported cases for import-flow edges")

            movement_counts = defaultdict(int)
            for _, row in imported_cases.iterrows():
                source = str(row['comuninf']).strip()
                target = str(row['id_mn_resi']).strip()
                if source and target and source != 'nan' and target != 'nan' and source != target:
                    movement_counts[(source, target)] += 1

            for (source, target), cnt in movement_counts.items():
                if source in municipalities and target in municipalities:
                    # distance: larger count → smaller distance
                    dist = 1.0 / (1.0 + cnt)
                    self.graph.add_edge(source, target, weight=dist, flow_type='import')
            logger.info(f"✓ Added {len(movement_counts):,} directed edges from import flow")

        # Temporal correlation edges (sparse top-K)
        logger.info("\n⚙️  Computing temporal correlations (sparse top-K per node)...")
        temporal_edges = self._add_sparse_temporal_edges(
            min_correlation=MIN_CORRELATION,
            topk=TOPK_TEMPORAL
        )
        logger.info(f"✓ Added {temporal_edges:,} temporal edges (sparse)")

        # Graph stats
        logger.info(f"\n📊 Network Statistics:")
        logger.info(f"   - Nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"   - Edges: {self.graph.number_of_edges():,}")
        if self.graph.number_of_nodes() > 0:
            avg_deg = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
            logger.info(f"   - Average degree: {avg_deg:.2f}")
            logger.info(f"   - Density: {nx.density(self.graph):.6f}")

        # SCC info
        if not nx.is_strongly_connected(self.graph):
            scc = list(nx.strongly_connected_components(self.graph))
            logger.info(f"   - Strongly connected components: {len(scc)}")
            if scc:
                logger.info(f"   - Largest SCC size: {len(max(scc, key=len)):,} nodes")

        return self.graph

    def _add_sparse_temporal_edges(self, min_correlation=0.7, topk=15):
        """
        Create sparse temporal correlation edges:
        - For each node, keep only its top-K correlated neighbors with corr >= threshold.
        - Weight is a distance: 1 - corr (smaller is stronger).
        """
        if 'sem_pri' not in self.df.columns:
            logger.warning("   ⚠️ No 'sem_pri' (epi week) data; skipping temporal correlations.")
            return 0

        logger.info(f"   Threshold corr ≥ {min_correlation}; Top-K per node = {topk}")
        # Build weekly counts per municipality
        muni_timeseries = {}
        for muni in self.graph.nodes():
            muni_cases = self.df[self.df['id_mn_resi'] == muni]
            if len(muni_cases) > 0:
                week_counts = muni_cases['sem_pri'].value_counts().sort_index()
                muni_timeseries[muni] = week_counts

        logger.info(f"   ✓ Built time series for {len(muni_timeseries):,} municipalities")

        # Pre-collect aligned union of weeks to vectorize per pair on-the-fly
        all_weeks = sorted(self.df['sem_pri'].dropna().unique().tolist()) if 'sem_pri' in self.df.columns else []
        week_index = {w: i for i, w in enumerate(all_weeks)}

        # Convert each muni series to a dense vector over all_weeks (optional tradeoff)
        # To save RAM, we’ll build vectors lazily per municipality and cache them.
        vec_cache = {}

        def vec_for(m):
            if m in vec_cache:
                return vec_cache[m]
            ts = muni_timeseries.get(m)
            if ts is None:
                v = np.zeros(len(all_weeks), dtype=float)
            else:
                v = np.zeros(len(all_weeks), dtype=float)
                # place counts
                for w, c in ts.items():
                    idx = week_index.get(w)
                    if idx is not None:
                        v[idx] = c
            vec_cache[m] = v
            return v

        municipalities = list(muni_timeseries.keys())
        edges_added = 0

        # For each node, compute correlations to others, keep top-K above threshold
        for i, m1 in enumerate(municipalities):
            v1 = vec_for(m1)
            if v1.std() == 0:
                continue  # no variation → no meaningful correlation

            # collect candidates
            cands = []
            for m2 in municipalities:
                if m2 == m1:
                    continue
                v2 = vec_for(m2)
                if v2.std() == 0:
                    continue
                # Pearson correlation
                corr = np.corrcoef(v1, v2)[0, 1]
                if np.isfinite(corr) and corr >= min_correlation:
                    cands.append((m2, float(corr)))

            if not cands:
                continue

            # top-K by correlation
            cands.sort(key=lambda x: x[1], reverse=True)
            for (m2, corr) in cands[:topk]:
                dist = 1.0 - corr   # distance in [0,1]; smaller = stronger
                # Add only one direction OR both as desired. We'll add one (m1->m2) to keep things sparser.
                if not self.graph.has_edge(m1, m2):
                    self.graph.add_edge(m1, m2, weight=dist, flow_type='temporal')
                    edges_added += 1

        return edges_added

    def compute_municipality_scores(self):
        """
        Compute a scalar function (risk/severity score) for each municipality.
        Score components:
        1. Case count (log)
        2. Severity proxies (hospitalization, death)
        3. Importation rate
        4. Trend (recent increase)
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: COMPUTING MUNICIPALITY RISK SCORES (Scalar Function)")
        logger.info("="*70)

        for muni in self.graph.nodes():
            muni_cases = self.df[self.df['id_mn_resi'] == muni]
            if len(muni_cases) == 0:
                self.municipality_scores[muni] = 0.0
                continue

            case_count = len(muni_cases)

            # Severity
            severity_score = 0.0
            if 'hospitaliz' in self.df.columns:
                hosp_rate = (muni_cases['hospitaliz'] == 1).sum() / len(muni_cases)
                severity_score += hosp_rate * 2.0
            if 'evolucao' in self.df.columns:
                death_rate = (muni_cases['evolucao'] == 2).sum() / len(muni_cases)
                severity_score += death_rate * 10.0

            # Importation
            import_score = 0.0
            if 'tpautocto' in self.df.columns:
                import_rate = (muni_cases['tpautocto'] == 2).sum() / len(muni_cases)
                import_score = import_rate * 1.5

            # Trend
            trend_score = 0.0
            if 'sem_pri' in self.df.columns and len(muni_cases) > 10:
                recent = muni_cases['sem_pri'].sort_values().tail(20)
                if len(recent) > 10:
                    first_half = recent.head(10).count()
                    second_half = recent.tail(10).count()
                    if first_half > 0:
                        trend_score = (second_half - first_half) / first_half

            score = (
                np.log1p(case_count) * 1.0 +
                severity_score * 0.8 +
                import_score * 0.6 +
                trend_score * 0.4
            )
            self.municipality_scores[muni] = score

        # Normalize to [0,1]
        max_score = max(self.municipality_scores.values()) if self.municipality_scores else 1.0
        if max_score > 0:
            self.municipality_scores = {k: v / max_score for k, v in self.municipality_scores.items()}

        logger.info(f"✓ Computed risk scores for {len(self.municipality_scores):,} municipalities")
        scores = list(self.municipality_scores.values())
        if scores:
            logger.info(f"\n📊 Score Stats | mean={np.mean(scores):.4f} std={np.std(scores):.4f} "
                        f"min={np.min(scores):.4f} max={np.max(scores):.4f} median={np.median(scores):.4f}")

            top_munis = sorted(self.municipality_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"\n🔝 Top 10 Municipalities by Risk Score:")
            for i, (muni, sc) in enumerate(top_munis, 1):
                case_count = len(self.df[self.df['id_mn_resi'] == muni])
                logger.info(f"   {i:2d}. {muni}: {sc:.4f} ({case_count:,} cases)")

        return self.municipality_scores

    def apply_discrete_morse_theory(self):
        """
        Apply discrete Morse theory to identify critical points in the transmission network.
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: APPLYING DISCRETE MORSE THEORY")
        logger.info("="*70)

        logger.info("⚙️  Computing discrete gradient vector field...")
        gradient_pairs = {}
        unpaired = set(self.graph.nodes())

        # Pair edges with largest score differences first (greedy)
        edges_sorted = sorted(
            self.graph.edges(),
            key=lambda e: abs(self.municipality_scores.get(e[0], 0) - self.municipality_scores.get(e[1], 0)),
            reverse=True
        )

        for u, v in edges_sorted:
            su = self.municipality_scores.get(u, 0)
            sv = self.municipality_scores.get(v, 0)
            if u not in gradient_pairs and v not in gradient_pairs:
                if su < sv:
                    gradient_pairs[u] = v
                    unpaired.discard(u); unpaired.discard(v)
                elif sv < su:
                    gradient_pairs[v] = u
                    unpaired.discard(u); unpaired.discard(v)

        logger.info(f"✓ Paired {len(gradient_pairs):,} node pairs")
        logger.info(f"✓ Unpaired nodes (critical candidates): {len(unpaired):,}")

        # Classify critical points
        self.critical_points = {'maxima': [], 'saddles': [], 'minima': []}
        for node in unpaired:
            score = self.municipality_scores.get(node, 0)
            neighbors = list(self.graph.successors(node)) + list(self.graph.predecessors(node))
            if not neighbors:
                continue
            neighbor_scores = [self.municipality_scores.get(n, 0) for n in neighbors]

            if score >= max(neighbor_scores):
                self.critical_points['maxima'].append(node)
            elif score <= min(neighbor_scores):
                self.critical_points['minima'].append(node)
            else:
                self.critical_points['saddles'].append(node)

        logger.info(f"\n📊 Critical Points:")
        logger.info(f"   - Maxima: {len(self.critical_points['maxima']):,}")
        logger.info(f"   - Saddles: {len(self.critical_points['saddles']):,}")
        logger.info(f"   - Minima: {len(self.critical_points['minima']):,}")

        return self.critical_points

    def identify_transmission_hubs(self, top_n=20):
        """
        Identify the most critical hubs by combining:
        - Saddle points (topology)
        - Maxima (potential sources)
        - Betweenness centrality (network importance; fast config)
        - Degree centrality
        - Risk score
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 5: IDENTIFYING CRITICAL TRANSMISSION HUBS (FAST)")
        logger.info("="*70)

        # Optionally restrict centrality to largest SCC for speed and better signal
        if FOCUS_ON_GIANT_SCC and self.graph.number_of_nodes() > 0:
            sccs = list(nx.strongly_connected_components(self.graph))
            if sccs:
                giant = max(sccs, key=len)
                Gc = self.graph.subgraph(giant).copy()
                logger.info(f"✓ Using largest SCC for centrality: {len(Gc):,} nodes")
            else:
                Gc = self.graph
        else:
            Gc = self.graph

        logger.info("⚙️  Computing betweenness centrality (approx)...")
        if USE_WEIGHTED_BETWEENNESS:
            # Weighted: uses Dijkstra; keep k small
            betweenness_sub = nx.betweenness_centrality(Gc, weight='weight', k=BETWEENNESS_K, seed=BETWEENNESS_SEED)
        else:
            # Unweighted: uses BFS; much faster and robust proxy
            betweenness_sub = nx.betweenness_centrality(Gc, weight=None, k=BETWEENNESS_K, seed=BETWEENNESS_SEED)

        # Extend to all nodes (0 for those outside Gc)
        betweenness = {n: 0.0 for n in self.graph.nodes()}
        betweenness.update(betweenness_sub)

        logger.info("⚙️  Computing degree centrality...")
        degree_centrality_sub = nx.degree_centrality(Gc)
        degree_centrality = {n: 0.0 for n in self.graph.nodes()}
        degree_centrality.update(degree_centrality_sub)

        logger.info("⚙️  Computing hub scores...")
        hub_scores = {}
        for node in self.graph.nodes():
            is_saddle = 1.5 if node in self.critical_points.get('saddles', []) else 0.0
            is_maxima = 1.2 if node in self.critical_points.get('maxima', []) else 0.0
            risk = self.municipality_scores.get(node, 0) * 1.0
            betw = betweenness.get(node, 0) * 2.0
            deg = degree_centrality.get(node, 0) * 1.0
            hub_scores[node] = is_saddle + is_maxima + risk + betw + deg

        top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        logger.info(f"✓ Identified top {top_n} hubs")

        logger.info("\n🎯 CRITICAL TRANSMISSION HUBS (Ranked):")
        logger.info("="*70)
        betw_vals = list(betweenness.values())
        p90 = np.percentile(betw_vals, 90) if betw_vals else 0.0
        for i, (muni, hs) in enumerate(top_hubs, 1):
            muni_data = self.df[self.df['id_mn_resi'] == muni]
            case_count = len(muni_data)
            risk_score = self.municipality_scores.get(muni, 0)
            betw_score = betweenness.get(muni, 0)
            degree = self.graph.degree(muni)

            hub_type = []
            if muni in self.critical_points['saddles']:
                hub_type.append("BRIDGE")
            if muni in self.critical_points['maxima']:
                hub_type.append("SOURCE")
            if betw_score > p90:
                hub_type.append("CONNECTOR")
            hub_type_str = "/".join(hub_type) if hub_type else "HUB"

            logger.info(f"\n{i:2d}. Municipality: {muni}")
            logger.info(f"    Hub Score: {hs:.4f}")
            logger.info(f"    Type: {hub_type_str}")
            logger.info(f"    Cases: {case_count:,}")
            logger.info(f"    Risk Score: {risk_score:.4f}")
            logger.info(f"    Betweenness: {betw_score:.6f}")
            logger.info(f"    Degree: {degree}")

        return top_hubs, hub_scores

    def generate_summary_report(self):
        """Generate a summary report of the analysis."""
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*70)

        logger.info(f"\n📋 Dataset Overview:")
        logger.info(f"   - Total cases analyzed: {len(self.df):,}")
        logger.info(f"   - Municipalities: {self.graph.number_of_nodes():,}")
        logger.info(f"   - Transmission pathways: {self.graph.number_of_edges():,}")

        logger.info(f"\n🔬 Morse Theory Results:")
        logger.info(f"   - Critical points: {sum(len(v) for v in self.critical_points.values()):,}")
        logger.info(f"   - Outbreak sources (maxima): {len(self.critical_points['maxima']):,}")
        logger.info(f"   - Transmission bridges (saddles): {len(self.critical_points['saddles']):,}")
        logger.info(f"   - Low-risk areas (minima): {len(self.critical_points['minima']):,}")

        logger.info(f"\n💡 Key Insights:")
        logger.info(f"   - Identified hubs are topologically and network-centrally critical.")
        logger.info(f"   - Monitoring hubs can provide early warning; intervening at bridges can disrupt spread.")

        logger.info(f"\n📝 Recommendations:")
        logger.info(f"   1. Prioritize surveillance in identified hubs.")
        logger.info(f"   2. Use hub metrics as features in ML forecasting.")
        logger.info(f"   3. Validate hubs against historical spread patterns.")
        logger.info(f"   4. Focus vector control on bridge municipalities.")

        logger.info("\n" + "="*70)
        logger.info("Analysis complete! See 'dengue_analysis.log' and output CSV.")
        logger.info("="*70 + "\n")

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("DENGUE TRANSMISSION NETWORK ANALYSIS (FAST)")
    print("Using Discrete Morse Theory to Identify Critical Hubs")
    print("="*70 + "\n")

    # ============================================================================
    # CONFIGURATION - CHANGE THIS TO YOUR FILE PATH
    # ============================================================================
    # DATA_SOURCE = "./DENGBR25.csv"
    DATA_SOURCE = "./DENGBR23.csv"
    # ============================================================================

    analyzer = DengueNetworkAnalyzer(DATA_SOURCE)

    try:
        analyzer.load_and_preprocess_data()
        analyzer.build_transmission_network()
        analyzer.compute_municipality_scores()
        analyzer.apply_discrete_morse_theory()
        top_hubs, hub_scores = analyzer.identify_transmission_hubs(top_n=20)
        analyzer.generate_summary_report()

        # Save results
        results_df = pd.DataFrame([
            {
                'municipality': muni,
                'hub_score': score,
                'risk_score': analyzer.municipality_scores.get(muni, 0),
                'is_saddle': muni in analyzer.critical_points.get('saddles', []),
                'is_maxima': muni in analyzer.critical_points.get('maxima', []),
                'degree': analyzer.graph.degree(muni)
            }
            for muni, score in top_hubs
        ])
        results_df.to_csv('transmission_hubs.csv', index=False)
        logger.info("✓ Saved results to 'transmission_hubs.csv'")

        print("\n✨ Analysis complete! Check the log file and CSV output for details.\n")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
