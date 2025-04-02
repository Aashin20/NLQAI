document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = 'http://127.0.0.1:8000';

    const csvFileInput = document.getElementById('csvFile');
    const uploadButton = document.getElementById('uploadButton');
    const tableSelect = document.getElementById('tableSelect');
    const schemaDisplay = document.getElementById('schemaDisplay');
    const columnList = document.getElementById('columnList');
    const naturalQueryInput = document.getElementById('naturalQuery');
    const generateSqlButton = document.getElementById('generateSqlButton');
    const sqlSection = document.getElementById('sqlSection');
    const generatedSqlTextarea = document.getElementById('generatedSql');
    const executeSqlButton = document.getElementById('executeSqlButton');
    const resultsSection = document.getElementById('resultsSection');
    const resultsTableContainer = document.getElementById('resultsTableContainer');
    const feedbackArea = document.getElementById('feedbackArea');
    const loader = document.getElementById('loader');
    const uploadPage = document.getElementById('uploadPage');
    const queryPage = document.getElementById('queryPage');
    const uploadedTablesList = document.getElementById('uploadedTablesList');
    const tablesListSection = document.getElementById('tablesListSection');

    let selectedTableName = null;
    let uploadedTableDetails = [];

    function showLoader() {
        loader.classList.remove('hidden');
    }

    function hideLoader() {
        loader.classList.add('hidden');
    }

    function displayFeedback(message, isError = false) {
        feedbackArea.textContent = message;
        feedbackArea.className = isError ? 'feedback error' : 'feedback success';
        setTimeout(() => {
             feedbackArea.textContent = '';
             feedbackArea.className = 'feedback';
        }, isError ? 8000 : 5000);
    }

    async function apiRequest(url, options = {}) {
        showLoader();
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                let errorMsg = `HTTP error! Status: ${response.status}`;
                try {
                    const errData = await response.json();
                    errorMsg = errData.detail || errorMsg;
                } catch (e) {}
                throw new Error(errorMsg);
            }
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.indexOf("application/json") !== -1) {
                return await response.json();
            } else {
                return {};
            }
        } catch (error) {
            console.error('API Request Error:', error);
            displayFeedback(`Error: ${error.message}`, true);
            throw error;
        } finally {
            hideLoader();
        }
    }

    function resetQueryState() {
        naturalQueryInput.value = '';
        generatedSqlTextarea.value = '';
        sqlSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        resultsTableContainer.innerHTML = '';
        executeSqlButton.disabled = true;
    }

    async function fetchTables() {
        try {
            const data = await apiRequest(`${API_BASE_URL}/tables`);
            uploadedTableDetails = data.tables || [];
            populateTableSelect(uploadedTableDetails);
            populateUploadedTablesList(uploadedTableDetails);
            if (uploadedTableDetails.length > 0) {
                 tableSelect.disabled = false;
                 tablesListSection.classList.remove('hidden');
            } else {
                 tableSelect.disabled = true;
                 tableSelect.innerHTML = '<option value="">-- No Tables Available --</option>';
                 tablesListSection.classList.add('hidden');
            }
        } catch (error) {
            tableSelect.disabled = true;
            tableSelect.innerHTML = '<option value="">-- Error Loading Tables --</option>';
            tablesListSection.classList.add('hidden');
        }
    }

    function populateTableSelect(tables) {
        tableSelect.innerHTML = '<option value="">-- Select a Table --</option>';
        tables.forEach(table => {
            const option = document.createElement('option');
            option.value = table.table_name;
            option.dataset.filename = table.filename;
            option.dataset.columns = JSON.stringify(table.columns);
            option.textContent = `${table.filename} (${table.table_name})`;
            tableSelect.appendChild(option);
        });
        if (selectedTableName && tableSelect.querySelector(`option[value="${selectedTableName}"]`)) {
            tableSelect.value = selectedTableName;
            handleTableSelect();
        } else {
             selectedTableName = null;
             schemaDisplay.classList.add('hidden');
             generateSqlButton.disabled = true;
             resetQueryState();
        }
    }

    function populateUploadedTablesList(tables) {
        uploadedTablesList.innerHTML = '';
        tables.forEach(table => {
            const li = document.createElement('li');
            li.textContent = `${table.filename} (Table: ${table.table_name})`;
            uploadedTablesList.appendChild(li);
        });
    }


    function displaySchema(columns) {
        columnList.innerHTML = '';
        if (columns && columns.length > 0) {
            columns.forEach(col => {
                const li = document.createElement('li');
                li.textContent = col;
                columnList.appendChild(li);
            });
            schemaDisplay.classList.remove('hidden');
        } else {
            schemaDisplay.classList.add('hidden');
        }
    }

     function populateResultsTable(columns, results) {
        resultsTableContainer.innerHTML = '';
        if (!results || results.length === 0) {
            resultsTableContainer.textContent = 'Query executed successfully, but returned no rows.';
            resultsSection.classList.remove('hidden');
            return;
        }

        const table = document.createElement('table');
        table.id = 'resultsTable';
        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        columns.forEach(colName => {
            const th = document.createElement('th');
            th.textContent = colName;
            headerRow.appendChild(th);
        });
        const tbody = table.createTBody();
        results.forEach(rowData => {
            const row = tbody.insertRow();
            columns.forEach(colName => {
                const cell = row.insertCell();
                cell.textContent = rowData[colName] !== null && rowData[colName] !== undefined ? rowData[colName] : 'NULL';
            });
        });

        resultsTableContainer.appendChild(table);
        resultsSection.classList.remove('hidden');
    }

    function handleUpload() {
        const file = csvFileInput.files[0];
        if (!file) {
            displayFeedback("Please select a CSV file first.", true);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        apiRequest(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData,
        }).then(result => {
            displayFeedback(`Successfully uploaded '${result.original_filename}' as table '${result.table_name}'.`);
            csvFileInput.value = '';
            fetchTables();
            showPage('queryPage');
        }).catch(error => {});
    }

    function handleTableSelect() {
        selectedTableName = tableSelect.value;
        resetQueryState();

        if (selectedTableName) {
            const selectedOption = tableSelect.options[tableSelect.selectedIndex];
            try {
                const columns = JSON.parse(selectedOption.dataset.columns || '[]');
                displaySchema(columns);
            } catch(e) {
                 console.error("Failed to parse columns from dataset", e);
                 displaySchema([]);
            }
            generateSqlButton.disabled = false;
        } else {
            schemaDisplay.classList.add('hidden');
            generateSqlButton.disabled = true;
        }
    }

    function handleGenerateSql() {
        const query = naturalQueryInput.value.trim();
        if (!query) {
            displayFeedback("Please enter your question first.", true);
            return;
        }
        if (!selectedTableName) {
            displayFeedback("Please select a table first.", true);
            return;
        }

        apiRequest(`${API_BASE_URL}/generate-sql`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                natural_language_query: query,
                table_name: selectedTableName
            })
        }).then(result => {
            generatedSqlTextarea.value = result.sql_query || '';
            sqlSection.classList.remove('hidden');
            executeSqlButton.disabled = !result.sql_query;
            resultsSection.classList.add('hidden');
            displayFeedback("SQL generated successfully!", false);
        }).catch(error => {
             generatedSqlTextarea.value = '';
             sqlSection.classList.add('hidden');
             executeSqlButton.disabled = true;
        });
    }

    function handleExecuteSql() {
        const sqlQuery = generatedSqlTextarea.value.trim();
         if (!sqlQuery) {
            displayFeedback("SQL query is empty.", true);
            return;
        }
        if (!selectedTableName) {
            displayFeedback("No table context selected.", true);
            return;
        }
         if (!sqlQuery.toLowerCase().startsWith('select')) {
              displayFeedback("Only SELECT queries are allowed.", true);
              return;
         }


        apiRequest(`${API_BASE_URL}/execute-sql`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sql_query: sqlQuery,
                table_name: selectedTableName
            })
        }).then(result => {
             populateResultsTable(result.columns || [], result.results || []);
             displayFeedback(result.message || "Query executed.", false);

        }).catch(error => {
            resultsSection.classList.add('hidden');
             resultsTableContainer.innerHTML = '';
        });
    }

    function showPage(pageId) {
        uploadPage.classList.add('hidden');
        queryPage.classList.add('hidden');

        if (pageId === 'uploadPage') {
            uploadPage.classList.remove('hidden');
        } else if (pageId === 'queryPage') {
            queryPage.classList.remove('hidden');
        }
    }

    uploadButton.addEventListener('click', handleUpload);
    tableSelect.addEventListener('change', handleTableSelect);
    generateSqlButton.addEventListener('click', handleGenerateSql);
    executeSqlButton.addEventListener('click', handleExecuteSql);

    fetchTables();
    showPage('uploadPage');
});