
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


// function styleReportTable() {
//     var tbl = document.getElementById('reportTable');
//     if (!tbl) return;

//     // Detect columns by header text
//     var headers = tbl.getElementsByTagName('th');
//     var numericCols = {};
//     for (var i = 0; i < headers.length; i++) {
//     var h = headers[i].innerText || headers[i].textContent;
//     if (/^(var|.*vol|cate(_suav.*)?|count|total)$/i.test(h)) numericCols[i] = 'num';
//     if (/%/.test(h) || /% ?var/i.test(h)) numericCols[i] = 'pct';
//     }

//     // Apply classes to cells
//     var rows = tbl.tBodies.length ? tbl.tBodies[0].rows : [];
//     for (var r = 0; r < rows.length; r++) {
//         var cells = rows[r].cells;
//         for (var c = 0; c < cells.length; c++) {
//             if (numericCols[c] === 'num') cells[c].className += ' num';
//             if (numericCols[c] === 'pct') cells[c].className += ' pct';
//         }
//     }
// }

