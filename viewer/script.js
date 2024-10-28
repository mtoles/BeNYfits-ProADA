
function displayJsonFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const lines = text.trim().split('\n');
        const conversationDiv = document.getElementById('conversation');
        conversationDiv.innerHTML = '';  // Clear previous content
        lines.forEach(line => {
            const json_line = JSON.parse(line);  // Parse each JSON line
            // Label header
            const labelContainerDiv = document.createElement('div');
            labelContainerDiv.classList.add('label-container');
            conversationDiv.appendChild(labelContainerDiv);

            // Label content
            const labels_dict = json_line['labels']; // {'label_name': 'label_value'}
            const predictions_dict = json_line['predictions'][json_line['predictions'].length - 1];
            const hh_diff = json_line['hh_diff'];

            // Acc table
            const allKeys = Array.from(new Set([...Object.keys(labels_dict), ...Object.keys(predictions_dict)]));

            // Create a table element
            const labelTable = document.createElement('table');
            labelTable.setAttribute('border', '1');
            labelTable.style.width = '100%';
            labelTable.classList.add('label-table');

            // Create table headers
            const header = labelTable.insertRow();
            const th1 = document.createElement('th');
            th1.innerText = "Key";
            const th2 = document.createElement('th');
            th2.innerText = "Label";
            const th3 = document.createElement('th');
            th3.innerText = "Prediction";
            header.appendChild(th1);
            header.appendChild(th2);
            header.appendChild(th3);

            // Populate the table rows
            allKeys.forEach(key => {
                const row = labelTable.insertRow();
                const cellKey = row.insertCell(0);
                const cellDict1 = row.insertCell(1);
                const cellDict2 = row.insertCell(2);

                cellKey.innerText = key;
                cellDict1.innerText = labels_dict[key] !== undefined ? labels_dict[key] : '';
                cellDict2.innerText = predictions_dict[key] !== undefined ? predictions_dict[key] : '';
            });

            // Append the table to the div
            const accTableDiv = document.createElement('div');
            accTableDiv.classList.add('acc-table');
            labelContainerDiv.appendChild(accTableDiv);
            accTableDiv.appendChild(labelTable);

            // Display the household non-defaults
            const hhDiffContainerDiv = document.createElement('div');
            hhDiffContainerDiv.classList.add('hh-diff-container');
            conversationDiv.appendChild(hhDiffContainerDiv);
            // hhDiffContainerDiv.style.whiteSpace = 'pre-wrap';
            hhDiffContainerDiv.textContent = hh_diff;

            json_line['dialog'].forEach(dialog => {
                // Previous messages
                const messages = dialog.map(message => message).slice(0, -1);
                const previousMessagesDiv = document.createElement('div');
                previousMessagesDiv.classList.add('previous-messages-container');
                previousMessagesDiv.classList.add('collapsed');  // Hide by default

                // Create chevron icon for collapsing
                const chevronDiv = document.createElement('div');
                chevronDiv.classList.add('chevron');
                chevronDiv.innerHTML = '&#9654;'
                chevronDiv.style.cursor = 'pointer';

                // Add toggle functionality for collapsing
                chevronDiv.addEventListener('click', function() {
                    const isCollapsed = previousMessagesDiv.classList.toggle('collapsed');
                    chevronDiv.innerHTML = isCollapsed ? '&#9654;' : '&#9660;';  // Toggle between right and down arrow
                });

                previousMessagesDiv.appendChild(chevronDiv);
                conversationDiv.appendChild(previousMessagesDiv);

                const messagesContainer = document.createElement('div');
                messagesContainer.classList.add('messages-content');
                messages.forEach(message => {
                    const roleDiv = document.createElement('div');  // RoleDiv
                    roleDiv.classList.add('role');
                    roleDiv.classList.add(message["role"]);
                    roleDiv.textContent = message["role"] + ": ";
                    messagesContainer.appendChild(roleDiv);
                    // Message
                    const messageDiv = document.createElement('div');
                    messageDiv.classList.add('message');
                    // messageDiv.style.whiteSpace = 'pre-wrap';
                    messageDiv.textContent = message["content"];
                    messagesContainer.appendChild(messageDiv);
                });
                previousMessagesDiv.appendChild(messagesContainer);

                // Last role
                const lastRoleDiv = document.createElement('div');  // RoleDiv
                lastRoleDiv.classList.add('role');
                const role = dialog[dialog.length - 1]["role"];
                lastRoleDiv.classList.add(role);
                lastRoleDiv.textContent = role + ": ";
                conversationDiv.appendChild(lastRoleDiv);
                // Last message
                const lastMessage = dialog[dialog.length - 1].content;
                const lastMessageDiv = document.createElement('div');
                lastMessageDiv.classList.add('message');
                lastMessageDiv.classList.add('last-message');
                lastMessageDiv.textContent = lastMessage;
                // lastMessageDiv.style.whiteSpace = 'pre-wrap';
                conversationDiv.appendChild(lastMessageDiv);
            });
        // Add a div for spacing
        const spacingDiv = document.createElement('div');
        spacingDiv.classList.add('spacing');
        conversationDiv.appendChild(spacingDiv);
        });
    };
    reader.readAsText(file);
}

// fetch('/download')
//   .then(response => response.blob())
//   .then(blob => {
//     const url = window.URL.createObjectURL(blob);
//     const a = document.createElement('a');
//     a.href = url;
//     a.download = 'example.pdf'; // Suggested filename
//     document.body.appendChild(a);
//     a.click();
//     a.remove();
//   })
//   .catch(err => console.error('Error fetching file:', err));


// by default, get the file from server.js port 3000 using fetch
// fetch('/latest').then(response => response.blob()).then(blob => displayJsonFile(blob));
// console.log('fetching latest file');
// fetch('/latest')
//   .then(response => response.text())
//   .then(text => displayJsonFile(text))
//   .catch(err => console.error('Error fetching file:', err));

document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;
    displayJsonFile(file);
});