
<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Inverted Indexer</h3>

  <p align="center">
  An Inverted Indexer written in Python
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <ul>
        <li><a href="#how-to-run">How to Run</a></li>
        <li><a href="#example-images">Example Images</a></li>
      </ul>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This creates an <a href="https://en.wikipedia.org/wiki/Inverted_index"><i>Inverted Index</i></a> for a given corpus. 
Inverted Index is a mapping of content (Words, Numbers etc) to its position in various documents. This speeds up query searches on the whole corpus. 


### Built With

* [Python](https://www.python.org)
* [NLTK](http://www.nltk.org)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python3](https://www.python.org/downloads/)
* [git](https://git-scm.com)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/saeenyoda/Inverted_Indexing.git
   ```
2. Install Requirements
   ```sh
   pip3 install nltk
   ```


<!-- USAGE EXAMPLES -->
## Usage

### How to Run
1. Open up command line or terminal and navigate to the cloned repo's directory
   ```sh
   cd "PATH-TO-DIRECTORY"
   ```
2. Place the blocks of your corpus in numbered sub-directories. 
   ```sh
   e.g. "PATH-TO-DIRECTORY/1"
   ```
3. Run the indexer.py file (use python if you have created it as an alias for python3)
   ```sh
   python3 indexer.py
   ```
    
    
<!-- LICENSE -->
## License

Distributed under the MIT License.



