
function renderTableFromJSON(tableOrId, data) {
  var table = (typeof tableOrId === "string")
                ? document.getElementById(tableOrId)
                : tableOrId;
  if (!table) { throw new Error("Table element not found"); }

  // Clear existing content
  while (table.firstChild) table.removeChild(table.firstChild);

  // Build THEAD
  var thead = document.createElement("thead");
  var trh = document.createElement("tr");
  for (var c = 0; c < data.columns.length; c++) {
    var th = document.createElement("th");
    // IE: use innerText for widest compatibility
    th.innerText = data.columns[c].label;
    trh.appendChild(th);
  }
  thead.appendChild(trh);
  table.appendChild(thead);

  // Build TBODY
  var tbody = document.createElement("tbody");
  var cols = data.columns; // keep order

  for (var i = 0; i < data.rows.length; i++) {
    var tr = document.createElement("tr");
    var row = data.rows[i];

    for (var c2 = 0; c2 < cols.length; c2++) {
      var key = cols[c2].key;
      var td = document.createElement("td");
      var val = row.hasOwnProperty(key) ? row[key] : "";

      // Render null/undefined as empty
      if (val === null || typeof val === "undefined") val = "";

      // Convert to string for cell
      td.innerText = (typeof val === "number") ? String(val) : String(val);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}
