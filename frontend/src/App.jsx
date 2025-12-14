import React, { useEffect, useMemo, useRef, useState } from 'react'
import { gsap } from 'gsap'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:5000'

export default function App() {
  const [url, setUrl] = useState('')
  const [company, setCompany] = useState('')
  const [fullname, setFullname] = useState('')
  const [email, setEmail] = useState('')
  const [csvFile, setCsvFile] = useState(null)
  const [renderJs, setRenderJs] = useState(true)
  const [dragging, setDragging] = useState(false)

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const [resultsFiles, setResultsFiles] = useState([])
  const [selectedFile, setSelectedFile] = useState('')
  const [resultsTable, setResultsTable] = useState(null)
  const [sortKey, setSortKey] = useState('website_url')
  const [sortDir, setSortDir] = useState('asc')
  const [filterText, setFilterText] = useState('')
  const [tableNotice, setTableNotice] = useState('')
  const [csvNotice, setCsvNotice] = useState('')

  const cardsRef = useRef([])
  const resultRef = useRef(null)

  useEffect(() => {
    async function loadFiles() {
      try {
        const response = await fetch(`${API_BASE}/results_list`)
        const files = await response.json()
        setResultsFiles(files)
        if (files.length > 0 && !selectedFile) {
          setSelectedFile(files[0])
        }
      } catch (err) {
        // optional endpoint; ignore failures
      }
    }
    loadFiles()
  }, [selectedFile])

  useEffect(() => {
    if (cardsRef.current.length) {
      gsap.fromTo(
        cardsRef.current,
        { opacity: 0, y: 24 },
        {
          opacity: 1,
          y: 0,
          duration: 0.7,
          ease: 'power2.out',
          stagger: 0.08,
          clearProps: 'opacity,transform'
        }
      )
    }
  }, [])

  useEffect(() => {
    if (resultRef.current) {
      gsap.fromTo(
        resultRef.current,
        { opacity: 0, y: 16 },
        { opacity: 1, y: 0, duration: 0.5, ease: 'power2.out', clearProps: 'opacity,transform' }
      )
    }
  }, [result])

  async function submit(e) {
    e.preventDefault()
    setLoading(true)
    setResult(null)

    try {
      if (csvFile) {
        const form = new FormData()
        form.append('file', csvFile)
        if (renderJs) form.append('render_js', '1')

        const res = await fetch(`${API_BASE}/analyze_csv`, {
          method: 'POST',
          body: form
        })
        const data = await res.json()
        setResult(data)
        if (Array.isArray(data?.results_preview)) {
          setResultsTable(data.results_preview)
          setTableNotice(`Loaded preview of ${data.results_preview.length} rows.`)
          // also attempt to fetch the full table once the backend writes it
          loadLatestTable()
        }
      } else {
        const res = await fetch(`${API_BASE}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url, company, fullname, email, render_js: renderJs })
        })
        const data = await res.json()
        setResult(data)
        if (data?.result) {
          setResultsTable([data.result])
          setTableNotice('Loaded latest single result.')
          loadLatestTable()
        }
      }
    } catch (err) {
      setResult({ error: err.message })
    }

    setLoading(false)
  }

  const getCompanyLabel = row =>
    row?.company ||
    row?.company_name ||
    row?.business ||
    row?.business_name ||
    row?.company_fullname ||
    row?.full_name ||
    ''

  const getEmailLabel = row => row?.email || row?.contact_email || row?.email_address || ''

  const filteredAndSorted = useMemo(() => {
    if (!resultsTable || !Array.isArray(resultsTable)) return []
    const ft = filterText.toLowerCase()
    return resultsTable
      .filter(row => {
        if (!ft) return true
        return (
          (row.website_url || '').toLowerCase().includes(ft) ||
          getCompanyLabel(row).toLowerCase().includes(ft) ||
          getEmailLabel(row).toLowerCase().includes(ft)
        )
      })
      .sort((a, b) => {
        const getSortValue = (row, key) => {
          if (key === 'company') {
            return getCompanyLabel(row).toLowerCase()
          }
          return (row[key] || '').toLowerCase()
        }
        const aVal = getSortValue(a, sortKey)
        const bVal = getSortValue(b, sortKey)
        if (aVal < bVal) return sortDir === 'asc' ? -1 : 1
        if (aVal > bVal) return sortDir === 'asc' ? 1 : -1
        return 0
      })
  }, [resultsTable, filterText, sortKey, sortDir])

  const summary = result?.result
  const isBatch = typeof result?.count === 'number'

async function handleDownloadCsv() {
  setCsvNotice('')
  try {
    const res = await fetch(`${API_BASE}/results.csv`)
    if (!res.ok) throw new Error('No results file available yet.')
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'results.csv'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      setCsvNotice('Downloaded latest CSV.')
    } catch (err) {
    setCsvNotice('No results available yet. Run an analysis first.')
  }
}

async function loadLatestTable() {
  try {
    const r = await fetch(`${API_BASE}/results.json`)
    if (!r.ok) throw new Error('No results file available yet.')
    const j = await r.json()
    setResultsTable(j)
    setTableNotice(j.length ? `Loaded ${j.length} rows.` : 'No results yet.')
  } catch (err) {
    setTableNotice(prev => {
      if (resultsTable && resultsTable.length > 0) {
        return 'Unable to refresh table right now. Showing preview.'
      }
      return 'No results available yet. Run an analysis first.'
    })
  }
}

async function handleFetchTable() {
  setTableNotice('')
  await loadLatestTable()
}

  function handleFileSelect(file) {
    if (file) setCsvFile(file)
  }

  function handleDrop(e) {
    e.preventDefault()
    e.stopPropagation()
    setDragging(false)
    const file = e.dataTransfer?.files?.[0]
    handleFileSelect(file)
  }

  function handleDrag(e, isOver) {
    e.preventDefault()
    e.stopPropagation()
    setDragging(isOver)
  }

  return (
    <div className="page">
      <div className="shell">
        <header className="hero fade">
          <div>
            <p className="eyebrow">Vision + Tech Signals</p>
            <h1>Website Analyzer</h1>
            <p className="lede">
              Send a single URL or a CSV batch. We render, capture, and score modern UX, chat presence, and friction points.
            </p>
            <div className="inline-flex gap-3 flex-wrap mt-3">
              <span className="pill">Playwright rendering</span>
              <span className="pill">Gemini vision</span>
              <span className="pill">CSV batches</span>
            </div>
          </div>
          <div className="stat-card fade" ref={el => (cardsRef.current[0] = el)}>
            <div className="stat-value">{resultsFiles.length || '0'}</div>
            <div className="stat-label">Saved result files</div>
            <div className="stat-sub">Download any run as CSV</div>
          </div>
        </header>

        <div className="grid-two">
          <div className="panel fade" ref={el => (cardsRef.current[1] = el)}>
            <div className="panel-head">
              <div>
                <p className="eyebrow">Single or Batch</p>
                <h2>Analyze a site</h2>
              </div>
              <div className="toggle">
                <label htmlFor="renderJs">Render JS</label>
                <input id="renderJs" type="checkbox" checked={renderJs} onChange={e => setRenderJs(e.target.checked)} />
              </div>
            </div>

            <form onSubmit={submit} className="stack">
              <label className="field">
                <span>Website URL</span>
                <input value={url} onChange={e => setUrl(e.target.value)} placeholder="https://example.com" />
              </label>

              <div className="grid-two-sm">
                <label className="field">
                  <span>Company</span>
                  <input value={company} onChange={e => setCompany(e.target.value)} placeholder="Acme Roofing" />
                </label>
                <label className="field">
                  <span>Full name</span>
                  <input value={fullname} onChange={e => setFullname(e.target.value)} placeholder="Jane Doe" />
                </label>
              </div>
              <label className="field">
                <span>Email (optional)</span>
                <input value={email} onChange={e => setEmail(e.target.value)} placeholder="name@company.com" />
              </label>

              <label className="field">
                <span>CSV upload (optional)</span>
                <div
                  className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
                  onDragEnter={e => handleDrag(e, true)}
                  onDragOver={e => handleDrag(e, true)}
                  onDragLeave={e => handleDrag(e, false)}
                  onDrop={handleDrop}
                >
                  <div className="dropzone-inner">
                    <span className="dropzone-icon">CSV</span>
                    <div className="dropzone-text">
                      <p className="muted">{csvFile ? `Selected: ${csvFile.name}` : 'Drag & drop CSV here or click to choose'}</p>
                      <small>CSV only – url, company, fullname, email (processed in chunks of 5)</small>
                    </div>
                  </div>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={e => handleFileSelect(e.target.files[0])}
                    aria-label="CSV upload"
                  />
                </div>
                <small>Columns: url, company, fullname, email. Free tier: processed in chunks of 5 rows.</small>
              </label>

              <button type="submit" disabled={loading} className="cta">
                {loading ? 'Analyzingâ€¦' : csvFile ? 'Run batch' : 'Analyze now'}
              </button>
            </form>
          </div>

          <div className="panel fade" ref={el => (cardsRef.current[2] = el)}>
              <div className="panel-head">
                <div>
                  <p className="eyebrow">Downloads</p>
                  <h2>Latest & history</h2>
                </div>
              </div>
              <div className="stack">
              <button type="button" className="ghost" onClick={handleDownloadCsv}>
                Download latest CSV
              </button>
              {csvNotice && <p className="muted">{csvNotice}</p>}
              <label className="field">
                <span>Pick a past run</span>
                <select value={selectedFile} onChange={e => setSelectedFile(e.target.value)}>
                  <option value="">-- select --</option>
                  {resultsFiles.map(f => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </label>
              {selectedFile && (
                <a className="link" href={`${API_BASE}/results/${selectedFile}`}>
                  Download selected
                </a>
              )}
              <button
                type="button"
                className="ghost"
                onClick={handleFetchTable}
              >
                Show latest results table
              </button>
              {tableNotice && <p className="muted">{tableNotice}</p>}
            </div>
          </div>
        </div>

        {result && (
          <div className="panel fade" ref={resultRef}>
            {result.error && <div className="error">{result.error}</div>}

            {summary && (
              <div className="stack-sm">
                <p className="eyebrow">Summary</p>
                <h3>{summary.website_url || url}</h3>
                <p className="muted">
                  Problems: <strong>{summary.problem || summary.problems_paragraph}</strong>
                </p>
                <p className="muted">
                  Offer: <strong>{summary.offer || summary.offers_paragraph}</strong>
                </p>
                <div className="inline-flex gap-2 flex-wrap">
                  {summary.errors && summary.errors.length > 0 && <span className="tag warning">Errors noted</span>}
                  {summary.model_used && <span className="tag">Model: {summary.model_used}</span>}
                  <a className="link" href={`${API_BASE}/results.csv`}>
                    Download CSV
                  </a>
                </div>
              </div>
            )}

            {isBatch && (
              <div className="stack-sm mt-4">
                <p className="eyebrow">Batch complete</p>
                <h3>{result.count} sites analyzed</h3>
                <p className="muted">Preview is available in the table below.</p>
              </div>
            )}
          </div>
        )}

        {resultsTable && resultsTable.length > 0 && (
          <div className="panel fade">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Results</p>
                <h2>Latest table</h2>
              </div>
              <div className="inline-flex gap-2">
                <input
                  value={filterText}
                  onChange={e => setFilterText(e.target.value)}
                  placeholder="Filter by website or company"
                />
                <select value={sortKey} onChange={e => setSortKey(e.target.value)}>
                  <option value="website_url">Website</option>
                  <option value="company">Company</option>
                </select>
                <button className="ghost" onClick={() => setSortDir(sortDir === 'asc' ? 'desc' : 'asc')}>
                  {sortDir === 'asc' ? 'Asc' : 'Desc'}
                </button>
              </div>
            </div>
             <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Website</th>
                    <th>Company</th>
                    <th>Email</th>
                    <th>Problem</th>
                    <th>Offer</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAndSorted.map((row, i) => (
                    <tr key={i}>
                      <td>{row.website_url}</td>
                      <td>{getCompanyLabel(row)}</td>
                      <td>{getEmailLabel(row)}</td>
                      <td>{row.problem || row.problems_paragraph}</td>
                      <td>{row.offer || row.offers_paragraph}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {resultsTable && resultsTable.length === 0 && (
          <div className="panel fade">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Results</p>
                <h2>Latest table</h2>
              </div>
            </div>
            <p className="muted">No results available yet. Run an analysis to populate the table.</p>
          </div>
        )}
      </div>
    </div>
  )
}



