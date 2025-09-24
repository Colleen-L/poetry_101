import {useState} from 'react';
import './App.css'

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    // obtain search results from backend
    if (!query.trim()) setResults([]);;

    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(query)}`);
      const data = await response.json();

      setResults(data.slice(0, 10)); // Get top 10 results
    } catch (error) {
      console.error("Error fetching search results:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="container">
        <h1>📜 Poetry 101</h1>
        <h3>The best tool for teachers and students to read and learn about poetry.</h3>
        <h3>This tool utilizes the e-book on Project Gutenbery, <i>Poems Teachers Ask For</i>, produced by Charles Aldarondo and the Online Distributed Proofreading Team. </h3>
        <br></br>
        <br></br>
        <div className="search-container">
          <input 
          type="search"
          className="search-bar"
          placeholder='🔍  What poem are you looking for?'
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(); }}></input>
          <button onClick={handleSearch}>Search</button>
        </div>

        {loading && <p>Loading...</p>}

        <ul>
          {results.map(({ title, author }, idx) => (
            <li key={idx}>
              <strong>{title}</strong> {author ? `by ${author}` : ""}
            </li>
          ))}
        </ul>
      </div>
    </>
  )
}

export default App
