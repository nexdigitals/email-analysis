from pathlib import Path
path = Path('frontend/src/App.jsx')
data = path.read_text()
old = "  try:\n    const r = await fetch(`${API_BASE}/results.json`)\n    if (!r.ok) throw new Error('No results file available yet.')\n    const j = await r.json()\n    setResultsTable(j)\n    setTableNotice(j.length ? `Loaded ${j.length} rows.` : 'No results yet.')\n  } catch (err) {\n    setResultsTable([])\n    setTableNotice('No results available yet. Run an analysis first.')\n  }\n}"
new = "  try:\n    const r = await fetch(`${API_BASE}/results.json`)\n    if (!r.ok) throw new Error('No results file available yet.')\n    const j = await r.json()\n    setResultsTable(j)\n    setTableNotice(j.length ? `Loaded ${j.length} rows.` : 'No results yet.')\n  } catch (err) {\n    setTableNotice(prev => prev || 'No results yet; preview is shown.')\n  }\n}"
if old not in data:
    raise SystemExit('pattern not found')
path.write_text(data.replace(old, new, 1))
