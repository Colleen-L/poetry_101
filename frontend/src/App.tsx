import { useEffect, useState} from 'react';
import './App.css'

const themes = [
  { name: "Childhood and Innocence"},
  { name: "Nature and New Beginnings"},
  { name: "Justice, Honor, and Humanity"},
  { name: "Rural life and Nature"},
  { name: "Patriotism and Youth"},
  { name: "Classical/Old Language"},
  { name: "Music and Romance"},
  { name: "Conflict and Labor"}
];

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  const handleSearch = async () => {
    // obtain search results from backend
    if (!query.trim()) setResults([]);;

    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(query)}`);
      const data = await response.json();

      setResults(data.slice(0, 10));
    } catch (error) {
      console.error("Error fetching search results:", error);
    } finally {
      setLoading(false);
    }
  };

  const [selectedTheme, setSelectedTheme] = useState<string | null>(null);

  const fetchByTheme = async (theme: string) => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/theme/${encodeURIComponent(theme)}`);
      const data = await response.json();
      setResults(data);
      setSelectedTheme(theme);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="container">
        <div className="sidebar-container">
          <div className="sidebar">
            <h3>Themes</h3>
            {themes.map(({ name }) => (
              <button
                key={name}
                className={`theme-button ${selectedTheme === name ? 'active' : ''}`}
                onClick={() => fetchByTheme(name)}
              >
                {name}
              </button>
            ))}
            {selectedTheme && (
              <button onClick={() => { setResults([]); setSelectedTheme(null); }} className="clear-button">
                Clear Theme
              </button>
            )}
          </div>
        </div>
        <div className="main-content">
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
            {results.map(({ title, author, poem }, idx) => {
              const isExpanded = expandedIndex === idx;

              return (
                <li
                  key={idx}
                  onClick={() => setExpandedIndex(isExpanded ? null : idx)}
                  style={{
                    cursor: "pointer",
                    marginBottom: '1em',
                    listStyle: 'none',
                    padding: '10px',
                    borderBottom: '1px solid #ddd'
                  }}
                  >
                  <strong>
                    {isExpanded ? "▼ " : "▶ "}
                    {title}
                  </strong>
                  {author ? ` by ${author}` : ""}
                  
                  {isExpanded && (
                    <div
                      style={{
                        marginTop: '0.5em',
                        whiteSpace: 'pre-wrap',
                        fontStyle: 'italic',
                        backgroundColor: 'rgba(255, 255, 255, 0.2)',
                        padding: '10px',
                        borderRadius: '5px'
                      }}
                      >
                      {poem}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </>
  )
}

export default App
