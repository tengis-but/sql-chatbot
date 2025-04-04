document.getElementById('chat-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const questionInput = document.getElementById('question');
    const question = questionInput.value.trim();
    if (!question) return;

    const chatDisplay = document.getElementById('chat-display');
    const userP = document.createElement('p');
    userP.className = 'user';
    userP.textContent = question;
    chatDisplay.appendChild(userP);

    const loader = document.getElementById('loading');
    loader.style.display = 'block';

    console.log('Clearing input...');
    questionInput.value = '';
    questionInput.focus();

    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'question=' + encodeURIComponent(question)
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        loader.style.display = 'none';
        console.log('Received data:', data);  // Debug log to verify data

        const aiP = document.createElement('p');
        aiP.className = 'ai';
        aiP.textContent = data.response;
        chatDisplay.appendChild(aiP);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;

        if (data.table) {
            console.log('Rendering table:', data.table);
            const tableOutput = document.getElementById('table-output');
            tableOutput.innerHTML = '';
            const table = document.createElement('table');
            const thead = document.createElement('thead');
            const tbody = document.createElement('tbody');
            const headerRow = document.createElement('tr');
            data.table.headers.forEach(header => {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            data.table.rows.forEach(row => {
                const tr = document.createElement('tr');
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = cell;
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            table.appendChild(thead);
            table.appendChild(tbody);
            tableOutput.appendChild(table);
            document.getElementById('export-btn').style.display = 'block';
        }

        if (data.graph) {
            console.log('Rendering graph:', data.graph);
            const graphOutput = document.getElementById('graph-output');
            graphOutput.innerHTML = `<img src="data:image/png;base64,${data.graph}" alt="Graph">`;
        }
    })
    .catch(error => {
        loader.style.display = 'none';
        console.error('Error:', error);
        const aiP = document.createElement('p');
        aiP.className = 'ai';
        aiP.textContent = 'Sorry, something went wrong!';
        chatDisplay.appendChild(aiP);
    });
});

document.getElementById('export-btn').addEventListener('click', function() {
    fetch('/export')
        .then(response => response.json())
        .then(data => alert(data.message))
        .catch(error => console.error('Export error:', error));
});
