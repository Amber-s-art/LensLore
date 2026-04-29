<br />
<div align="center">
  <a href="https://github.com/Amber-s-art/LensLore">
    <img src="https://via.placeholder.com/150x150?text=LensLore+Logo" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">LensLore 🎞️✨</h3>

  <p align="center">
    <strong>Lens = Cinema. Lore = Knowledge.</strong><br>
    An intelligent cinematic discovery engine for Bollywood and Hollywood.
    <br />
    <br />
    <a href="https://github.com/Amber-s-art/LensLore"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Amber-s-art/LensLore">View Demo</a>
    ·
    <a href="https://github.com/Amber-s-art/LensLore/issues">Report Bug</a>
    ·
    <a href="https://github.com/Amber-s-art/LensLore/issues">Request Feature</a>
  </p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/API-TMDB-01b4e4.svg" alt="TMDB">
</div>

---

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#tech-stack">Tech Stack</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

---

## 📖 About The Project

[![LensLore Screen Shot][product-screenshot]](https://github.com/Amber-s-art/LensLore)

Welcome to **LensLore**—where cinema meets data. LensLore is an intelligent cinematic discovery engine tailored for both Bollywood and Hollywood. Moving beyond basic filters, it leverages **TF-IDF** and **Cosine Similarity** to surface highly accurate film recommendations based on deep thematic tags, genres, and cast members. 

The entire experience is wrapped in a premium, editorial-style interface that makes exploring movies as captivating as watching them.

### ✨ Key Features
* 🌍 **Dual-Industry Database:** Seamlessly toggle between Bollywood and Hollywood cinematic universes.
* 🧠 **Smart Content-Based Filtering:** Analyzes tags, genres, overviews, and cast to find the most conceptually similar films.
* 🎭 **Deep Customization:** Filter the universe of films by specific genres or your favorite actors.
* 🎬 **Interactive Cinematic UI:** A custom Streamlit interface with a running film strip, glowing animations, and interactive movie cards with blur-on-hover synopsis reveals.
* 🔗 **Smart Media Links:** Automatically fetches real-time data via the **TMDB API**, prioritizing YouTube trailers, then official homepages.
* 📊 **Silent Analytics Logging:** Tracks user queries in a hidden backend CSV (`logs/recom.csv`) for future model evaluation.

---

## 🛠️ Tech Stack

This project is built using the following open-source technologies:

* **[Python 3](https://www.python.org/)** - Core logic and data processing
* **[Streamlit](https://streamlit.io/)** - Frontend framework (with custom CSS/HTML injection)
* **[Scikit-Learn](https://scikit-learn.org/)** - Machine Learning (TF-IDF Vectorization, NearestNeighbors)
* **[Pandas](https://pandas.pydata.org/)** - Data manipulation
* **[TMDB API](https://developer.themoviedb.org/docs)** - Live posters, ratings, and trailers

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites
Make sure you have Python installed on your machine. You will also need a TMDB API Read Access Token.
* Create a free account at [TMDB](https://www.themoviedb.org/) and navigate to Settings > API to generate your token.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Amber-s-art/LensLore.git](https://github.com/Amber-s-art/LensLore.git)
   cd LensLore
Create a virtual environment (Recommended):

Bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

Bash
pip install streamlit pandas scikit-learn requests
Add your TMDB API Key:
Open app.py and replace the placeholder TMDB_HEADERS token with your own Bearer token.

Ensure the datasets are in place:
Verify that dataset/cleaned/bollywood_cleaned.csv and dataset/cleaned/hollywood_cleaned.csv exist.

Run the application:

Bash
streamlit run app.py
💻 Usage
Open http://localhost:8501 in your browser.

Select your preferred industry (Bollywood or Hollywood).

Filter the catalog by Genre or Actor.

Select a specific movie from the dropdown.

Click "Discover Films" to generate 5 highly correlated recommendations based on the movie's thematic tags.

Hover over any movie card to read its synopsis, or click the card to watch the trailer.

📁 Repository Structure
Plaintext
LensLore/
│
├── app.py                     # Main Streamlit application and UI
├── dataset/
│   └── cleaned/
│       ├── bollywood_cleaned.csv  # Pre-processed Bollywood dataset
│       └── hollywood_cleaned.csv  # Pre-processed Hollywood dataset
├── logs/
│   └── recom.csv              # Auto-generated backend recommendation logs
└── README.md                  # Project documentation
🔮 Roadmap
[x] Initial release with TF-IDF and Cosine Similarity

[x] Implement TMDB API for live posters and trailers

[x] Custom CSS for premium UI

[ ] Collaborative Filtering: Integrate user-rating data to suggest films based on similar user profiles.

[ ] Data Pipeline Automation: Automatically pull in newly released movies weekly.

[ ] Analytics Dashboard: Create an admin page to visualize data from logs/recom.csv.

See the open issues for a full list of proposed features (and known issues).

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
Distributed under the MIT License. See LICENSE for more information.
