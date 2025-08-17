"""
Multimodal clustering analysis to identify distinct cell types/modalities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from skimage import measure, feature, filters
from scipy import ndimage
import logging

from config.analysis_config import ANALYSIS_CONFIG, IMAGE_CONFIG

class MultimodalAnalyzer:
    """Analyzes and clusters images based on cellular characteristics to identify modalities"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = ANALYSIS_CONFIG['clustering']
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_names = []
        
    def extract_features(self, image: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """
        Extract comprehensive features for modality clustering
        
        Args:
            image: Original RGB image
            ground_truth: Ground truth segmentation
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract different types of features
        if 'morphological' in self.config['feature_types']:
            features.update(self._extract_morphological_features(ground_truth))
            
        if 'texture' in self.config['feature_types']:
            features.update(self._extract_texture_features(image))
            
        if 'spatial' in self.config['feature_types']:
            features.update(self._extract_spatial_features(ground_truth))
        
        return features
    
    def _extract_morphological_features(self, ground_truth: np.ndarray) -> Dict:
        """Extract morphological features related to cell shape and size"""
        # Get cell regions (foreground)
        cell_mask = (ground_truth > 0).astype(int)
        labeled_cells, num_cells = ndimage.label(cell_mask)
        
        if num_cells == 0:
            return {
                'num_cells': 0,
                'mean_cell_area': 0,
                'std_cell_area': 0,
                'mean_cell_perimeter': 0,
                'mean_cell_circularity': 0,
                'mean_cell_solidity': 0,
                'mean_cell_aspect_ratio': 1,
                'cell_area_cv': 0
            }
        
        # Analyze individual cells
        cell_areas = []
        cell_perimeters = []
        cell_circularities = []
        cell_solidities = []
        cell_aspect_ratios = []
        
        for cell_id in range(1, num_cells + 1):
            cell_region = (labeled_cells == cell_id)
            
            # Basic measurements
            area = np.sum(cell_region)
            cell_areas.append(area)
            
            # Get region properties
            props = measure.regionprops(cell_region.astype(int))
            if len(props) > 0:
                prop = props[0]
                
                # Perimeter
                perimeter = prop.perimeter
                cell_perimeters.append(perimeter)
                
                # Circularity (4π*area/perimeter²)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                cell_circularities.append(circularity)
                
                # Solidity (area/convex_area)
                solidity = prop.solidity
                cell_solidities.append(solidity)
                
                # Aspect ratio (major_axis/minor_axis)
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1
                cell_aspect_ratios.append(aspect_ratio)
        
        return {
            'num_cells': num_cells,
            'mean_cell_area': np.mean(cell_areas),
            'std_cell_area': np.std(cell_areas),
            'mean_cell_perimeter': np.mean(cell_perimeters) if cell_perimeters else 0,
            'mean_cell_circularity': np.mean(cell_circularities) if cell_circularities else 0,
            'mean_cell_solidity': np.mean(cell_solidities) if cell_solidities else 0,
            'mean_cell_aspect_ratio': np.mean(cell_aspect_ratios) if cell_aspect_ratios else 1,
            'cell_area_cv': np.std(cell_areas) / np.mean(cell_areas) if np.mean(cell_areas) > 0 else 0
        }
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """Extract texture features from the original image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = np.mean(image, axis=2)
        else:
            gray_image = image
        
        # Normalize image
        gray_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min())
        
        # GLCM features
        try:
            glcm = feature.greycomatrix(
                (gray_image * 255).astype(np.uint8),
                distances=[1, 2, 3],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            contrast = feature.greycoprops(glcm, 'contrast').mean()
            dissimilarity = feature.greycoprops(glcm, 'dissimilarity').mean()
            homogeneity = feature.greycoprops(glcm, 'homogeneity').mean()
            energy = feature.greycoprops(glcm, 'energy').mean()
            correlation = feature.greycoprops(glcm, 'correlation').mean()
            
        except:
            # Fallback values if GLCM computation fails
            contrast = 0
            dissimilarity = 0
            homogeneity = 1
            energy = 1
            correlation = 1
        
        # Local Binary Pattern
        try:
            lbp = feature.local_binary_pattern(gray_image, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
            lbp_features = {f'lbp_bin_{i}': lbp_hist[i] for i in range(len(lbp_hist))}
        except:
            lbp_features = {f'lbp_bin_{i}': 0 for i in range(10)}
        
        # Basic intensity statistics
        intensity_features = {
            'mean_intensity': np.mean(gray_image),
            'std_intensity': np.std(gray_image),
            'min_intensity': np.min(gray_image),
            'max_intensity': np.max(gray_image),
            'intensity_range': np.max(gray_image) - np.min(gray_image)
        }
        
        # Combine all texture features
        texture_features = {
            'glcm_contrast': contrast,
            'glcm_dissimilarity': dissimilarity,
            'glcm_homogeneity': homogeneity,
            'glcm_energy': energy,
            'glcm_correlation': correlation,
            **lbp_features,
            **intensity_features
        }
        
        return texture_features
    
    def _extract_spatial_features(self, ground_truth: np.ndarray) -> Dict:
        """Extract spatial distribution features"""
        cell_mask = (ground_truth > 0).astype(int)
        labeled_cells, num_cells = ndimage.label(cell_mask)
        
        if num_cells == 0:
            return {
                'cell_density': 0,
                'nearest_neighbor_mean': 0,
                'nearest_neighbor_std': 0,
                'spatial_dispersion': 0,
                'clustering_coefficient': 0
            }
        
        # Cell density
        image_area = ground_truth.shape[0] * ground_truth.shape[1]
        cell_density = num_cells / image_area
        
        # Get cell centroids
        centroids = []
        for cell_id in range(1, num_cells + 1):
            cell_region = (labeled_cells == cell_id)
            centroid = ndimage.center_of_mass(cell_region)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        if len(centroids) < 2:
            return {
                'cell_density': cell_density,
                'nearest_neighbor_mean': 0,
                'nearest_neighbor_std': 0,
                'spatial_dispersion': 0,
                'clustering_coefficient': 0
            }
        
        # Nearest neighbor distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(centroids))
        
        # For each cell, find distance to nearest neighbor
        nn_distances = []
        for i in range(len(centroids)):
            # Get distances to all other cells (excluding self)
            cell_distances = distances[i]
            cell_distances = cell_distances[cell_distances > 0]  # Remove self-distance (0)
            if len(cell_distances) > 0:
                nn_distances.append(np.min(cell_distances))
        
        # Spatial dispersion (coefficient of variation of centroid coordinates)
        spatial_dispersion = np.std(centroids.flatten()) / np.mean(centroids.flatten()) if np.mean(centroids.flatten()) > 0 else 0
        
        # Simple clustering coefficient (fraction of cells with neighbors within threshold)
        threshold_distance = np.sqrt(image_area / num_cells)  # Expected distance in uniform distribution
        clustered_cells = sum(1 for d in nn_distances if d < threshold_distance)
        clustering_coefficient = clustered_cells / num_cells if num_cells > 0 else 0
        
        return {
            'cell_density': cell_density,
            'nearest_neighbor_mean': np.mean(nn_distances) if nn_distances else 0,
            'nearest_neighbor_std': np.std(nn_distances) if nn_distances else 0,
            'spatial_dispersion': spatial_dispersion,
            'clustering_coefficient': clustering_coefficient
        }
    
    def fit_clustering(self, feature_data: pd.DataFrame) -> Dict:
        """
        Fit K-means clustering to identify modalities
        
        Args:
            feature_data: DataFrame with features for all images
            
        Returns:
            Clustering results and metrics
        """
        # Prepare feature matrix
        feature_matrix = feature_data.select_dtypes(include=[np.number]).values
        self.feature_names = feature_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0, posinf=1, neginf=-1)
        
        # Standardize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Fit K-means clustering
        self.kmeans = KMeans(
            n_clusters=self.config['n_clusters'],
            random_state=self.config['random_state'],
            n_init=10
        )
        
        cluster_labels = self.kmeans.fit_predict(feature_matrix_scaled)
        
        # Compute clustering metrics
        silhouette_avg = silhouette_score(feature_matrix_scaled, cluster_labels)
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(feature_data, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'cluster_centers': self.kmeans.cluster_centers_,
            'cluster_analysis': cluster_analysis,
            'feature_importance': self._compute_feature_importance(feature_matrix_scaled, cluster_labels)
        }
    
    def _analyze_clusters(self, feature_data: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        """Analyze characteristics of each cluster"""
        cluster_analysis = {}
        
        for cluster_id in range(self.config['n_clusters']):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_data = feature_data[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Compute cluster statistics
            cluster_stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(feature_data) * 100,
                'feature_means': cluster_data.select_dtypes(include=[np.number]).mean().to_dict(),
                'feature_stds': cluster_data.select_dtypes(include=[np.number]).std().to_dict()
            }
            
            # Identify distinguishing characteristics
            distinguishing_features = self._find_distinguishing_features(
                cluster_data.select_dtypes(include=[np.number]),
                feature_data.select_dtypes(include=[np.number])
            )
            
            cluster_stats['distinguishing_features'] = distinguishing_features
            cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats
        
        return cluster_analysis
    
    def _find_distinguishing_features(self, cluster_data: pd.DataFrame, all_data: pd.DataFrame) -> List[Dict]:
        """Find features that distinguish this cluster from others"""
        distinguishing = []
        
        for feature in cluster_data.columns:
            cluster_mean = cluster_data[feature].mean()
            overall_mean = all_data[feature].mean()
            overall_std = all_data[feature].std()
            
            if overall_std > 0:
                z_score = abs(cluster_mean - overall_mean) / overall_std
                if z_score > 1.5:  # Significantly different
                    distinguishing.append({
                        'feature': feature,
                        'cluster_mean': cluster_mean,
                        'overall_mean': overall_mean,
                        'z_score': z_score,
                        'direction': 'higher' if cluster_mean > overall_mean else 'lower'
                    })
        
        # Sort by z-score (most distinguishing first)
        distinguishing.sort(key=lambda x: x['z_score'], reverse=True)
        return distinguishing[:5]  # Top 5 distinguishing features
    
    def _compute_feature_importance(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> Dict:
        """Compute feature importance for clustering"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a classifier to predict clusters based on features
        rf = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
        rf.fit(feature_matrix, cluster_labels)
        
        # Get feature importance
        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            importance_dict[feature_name] = rf.feature_importances_[i]
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'feature_importance': dict(sorted_importance),
            'top_features': [item[0] for item in sorted_importance[:10]]
        }
    
    def predict_modality(self, features: Dict) -> int:
        """Predict modality for a new image based on its features"""
        if self.kmeans is None:
            raise ValueError("Clustering model not fitted. Call fit_clustering first.")
        
        # Convert features to array
        feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        
        # Handle missing values
        feature_vector = np.nan_to_num(feature_vector, nan=0, posinf=1, neginf=-1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict cluster
        cluster_label = self.kmeans.predict(feature_vector_scaled)[0]
        
        return cluster_label
    
    def assign_modality_names(self, cluster_analysis: Dict) -> Dict:
        """Assign descriptive names to modalities based on their characteristics"""
        modality_names = {}
        
        for cluster_id in range(self.config['n_clusters']):
            cluster_key = f'cluster_{cluster_id}'
            if cluster_key not in cluster_analysis:
                continue
            
            cluster_info = cluster_analysis[cluster_key]
            distinguishing = cluster_info['distinguishing_features']
            
            # Determine modality type based on distinguishing features
            name = self._determine_modality_name(distinguishing, cluster_info['feature_means'])
            modality_names[cluster_id] = name
        
        return modality_names
    
    def _determine_modality_name(self, distinguishing_features: List[Dict], feature_means: Dict) -> str:
        """Determine descriptive name for modality based on characteristics"""
        # Analyze key characteristics
        num_cells = feature_means.get('num_cells', 0)
        mean_area = feature_means.get('mean_cell_area', 0)
        cell_density = feature_means.get('cell_density', 0)
        
        # Simple heuristic naming based on cell characteristics
        if num_cells < 20:
            if mean_area > 500:
                return "Large Sparse Cells"
            else:
                return "Small Sparse Cells"
        elif num_cells > 100:
            if mean_area < 200:
                return "Dense Small Cells"
            else:
                return "Dense Mixed Cells"
        else:
            if mean_area > 400:
                return "Medium Large Cells"
            else:
                return "Medium Small Cells"
