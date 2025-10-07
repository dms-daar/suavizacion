

function parseIntList(str) {
  if (str == null) return [];
  var parts = String(str).split(',');
  var out = [];
  for (var i = 0; i < parts.length; i++) {
    // trim (IE8-safe)
    var s = parts[i].replace(/^\s+|\s+$/g, '');
    if (s === '') continue;              // skip empty
    var n = parseInt(s, 10);             // radix 10
    if (!isNaN(n)) out.push(n);          // skip non-numeric
  }
  return out;
}

