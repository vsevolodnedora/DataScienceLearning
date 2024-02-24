# Project:

[HPI GitHub](https://github.com/KISZ-BB/image-dataset-curation-workshops/blob/main/notebooks/Tutorial_Part2_Downloading_Images_to_Google_Drive_and_Creating_Embeddings.ipynb)


### Workshop 1;

Google Notebooks:
- Copy of Tutorial_Part2_Image_Similarity_of_Street_Artwork.ipynb
- Copy of Tutorial_Part1_Downloading_Images_to_Google_Drive_and_Creating_Embeddings.ipynb
- Copy of Download_Images_from_Bing_to_Google_Drive.ipynb

- Downloading Paintings to Google Drive and Creating Image Embeddings

- Download images using `bing-image-downloader` library 
- Create embeddings using `https://github.com/christiansafka/img2vec` by Christian Safka writting with `PyTorch`
    - __Important__: images must be rescaled and renormalized in accordance with training data used for the pre-trained model. We used `resnet-34`. 
- Perform Image Similarity Analythsis
    - After the embeddings were created we performed the similarity analysis by computing the `cosine similarity`
    "Visualizing pairs of images can provide insights into the similarities and differences between them. This step often helps in understanding the characteristics of the images and how they relate to each other."
    - Next, we examined how to construct "Clusters of the Most Similar Images" For this we employed `NearestNeighbors` from `sklearn.neighbors`
    "we applyy clustering techniques to group images that are similar to each other. This can help in understanding the relationships between different images and identifying patterns within the dataset. In this case, we are interested in finding which are the most representative artworks in our dataset."
    - Then we "visualize the 9 nearest neighbors of every image in the dataset"
    - Finally we performed __Clustering Images__ employing `KMedoids` (Partitioning Around Medoid) from `sklearn_extra.cluster`. It is a classical partitioning technique of clustering that splits the data set of n objects into k clusters, where the number k of clusters assumed known a priori. 
    - Here we obtained _representative images_, the cluster centers, and visualized every neighborhood of several representative images
    - In this technique it is crucual to choose an 'Optimal' number of clusters. We eomplyed an `Elbow Method`, where we inspect the relation between the number of clusters and two properties: Distortion and Inertia. Here _Distortion_ is calculated as the average of the squared distances from the cluster centers of the respective clusters and _Inertia_ is calculated as the sum of squared distances of samples to their closest cluster center. 
    " Both of these methods usually give similar results. Both plots should show a decrease in intertia or distortion as we increase the number of clusters. We use the soft rule of choosing the number of clusters where we see "an elbow" (a big inflection) in both of these curves. "
    " In this set of plots, we see that the biggest inflection point happens at around 20 clusters, so it might be "good enough" to use 20 images to describe the dataset. Think about this not as a hard rule, but as a heuristic to simplify and reduce the time of your explorations. "