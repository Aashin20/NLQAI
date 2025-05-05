document.addEventListener('DOMContentLoaded', function() {
    const API_BASE_URL = 'http://localhost:8000';
    const UPLOAD_ENDPOINT = `${API_BASE_URL}/upload`;
    const TABLES_ENDPOINT = `${API_BASE_URL}/tables`;
    const GENERATE_SQL_ENDPOINT = `${API_BASE_URL}/generate-sql`;
    const EXECUTE_SQL_ENDPOINT = `${API_BASE_URL}/execute-sql`;

    const uploadPage = document.getElementById('uploadPage');
    const queryPage = document.getElementById('queryPage');
    const csvFileInput = document.getElementById('csvFile');
    const uploadButton = document.getElementById('uploadButton');
    const tablesListSection = document.getElementById('tablesListSection');
    const uploadedTablesList = document.getElementById('uploadedTablesList');
    const primaryTableSelect = document.getElementById('primaryTableSelect');
    const additionalTablesSelect = document.getElementById('additionalTablesSelect');
    const primarySchemaDisplay = document.getElementById('primarySchemaDisplay');
    const primaryColumnList = document.getElementById('primaryColumnList');
    const additionalSchemaDisplay = document.getElementById('additionalSchemaDisplay');
    const additionalColumnsContainer = document.getElementById('additionalColumnsContainer');
    const naturalQueryInput = document.getElementById('naturalQuery');
    const generateSqlButton = document.getElementById('generateSqlButton');
    const sqlSection = document.getElementById('sqlSection');
    const generatedSqlTextarea = document.getElementById('generatedSql');
    const executeSqlButton = document.getElementById('executeSqlButton');
    const resultsSection = document.getElementById('resultsSection');
    const resultsTableContainer = document.getElementById('resultsTableContainer');
    const feedbackArea = document.getElementById('feedbackArea');
    const loader = document.getElementById('loader');

    let tables = [];
    let tableSchemas = {};

    initApp();

    window.showPage = function(pageId) {
        document.querySelectorAll('.page').forEach(page => {
            page.classList.add('hidden');
        });
        document.getElementById(pageId).classList.remove('hidden');
    };

    function initApp() {
        loadTables();
        setupEventListeners();
    }

    function setupEventListeners() {
        uploadButton.addEventListener('click', handleFileUpload);
        primaryTableSelect.addEventListener('change', handlePrimaryTableChange);
        additionalTablesSelect.addEventListener('change', handleAdditionalTablesChange);
        generateSqlButton.addEventListener('click', generateSqlQuery);
        executeSqlButton.addEventListener('click', executeSqlQuery);
        naturalQueryInput.addEventListener('input', updateGenerateButtonState);
    }

    function loadTables() {
        showLoader();
        fetch(TABLES_ENDPOINT)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                tables = data.tables;
                populateTableSelections();
                updateTablesList();
                hideLoader();
            })
            .catch(error => {
                console.error('Error loading tables:', error);
                showFeedback('Error loading tables. Please check the API connection.', 'error');
                hideLoader();
            });
    }

    function populateTableSelections() {
        primaryTableSelect.innerHTML = '<option value="">-- Select a Primary Table --</option>';
        additionalTablesSelect.innerHTML = '';
        
        if (tables.length === 0) {
            showFeedback('No tables available. Please upload a CSV file first.', 'error');
            primaryTableSelect.disabled = true;
            additionalTablesSelect.disabled = true;
            return;
        }

        primaryTableSelect.disabled = false;
        additionalTablesSelect.disabled = false;
        
        tables.forEach(table => {
            const primaryOption = document.createElement('option');
            primaryOption.value = table.table_name;
            primaryOption.textContent = `${table.filename} (${table.table_name})`;
            primaryTableSelect.appendChild(primaryOption);
            
            const additionalOption = document.createElement('option');
            additionalOption.value = table.table_name;
            additionalOption.textContent = `${table.filename} (${table.table_name})`;
            additionalTablesSelect.appendChild(additionalOption);
            
            tableSchemas[table.table_name] = table.columns;
        });
    }

    function updateTablesList() {
        if (tables.length > 0) {
            uploadedTablesList.innerHTML = '';
            tables.forEach(table => {
                const listItem = document.createElement('li');
                listItem.textContent = `${table.filename} (Table: ${table.table_name}) - ${table.columns.length} columns`;
                uploadedTablesList.appendChild(listItem);
            });
            tablesListSection.classList.remove('hidden');
        } else {
            tablesListSection.classList.add('hidden');
        }
    }

    function handleFileUpload() {
        const file = csvFileInput.files[0];
        if (!file) {
            showFeedback('Please select a CSV file to upload.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        showLoader();
        fetch(UPLOAD_ENDPOINT, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(error => {
                    throw new Error(error.detail || 'Upload failed');
                });
            }
            return response.json();
        })
        .then(data => {
            showFeedback(`File "${data.original_filename}" uploaded successfully as table "${data.table_name}".`, 'success');
            csvFileInput.value = '';
            loadTables();
        })
        .catch(error => {
            console.error('Error uploading file:', error);
            showFeedback(`Upload failed: ${error.message}`, 'error');
            hideLoader();
        });
    }

    function handlePrimaryTableChange() {
        const selectedTable = primaryTableSelect.value;
        
        if (selectedTable) {
            displayTableSchema(selectedTable, primaryColumnList);
            primarySchemaDisplay.classList.remove('hidden');
            updateAdditionalTablesOptions(selectedTable);
            additionalTablesSelect.value = '';
            additionalSchemaDisplay.classList.add('hidden');
            additionalColumnsContainer.innerHTML = '';
        } else {
            primarySchemaDisplay.classList.add('hidden');
            primaryColumnList.innerHTML = '';
        }
        
        updateGenerateButtonState();
    }

    function updateAdditionalTablesOptions(primaryTable) {
        const currentSelection = Array.from(additionalTablesSelect.selectedOptions).map(opt => opt.value);
        
        additionalTablesSelect.innerHTML = '';
        
        tables.forEach(table => {
            if (table.table_name !== primaryTable) {
                const option = document.createElement('option');
                option.value = table.table_name;
                option.textContent = `${table.filename} (${table.table_name})`;
                option.selected = currentSelection.includes(table.table_name);
                additionalTablesSelect.appendChild(option);
            }
        });
    }

    function handleAdditionalTablesChange() {
        const selectedTables = Array.from(additionalTablesSelect.selectedOptions).map(opt => opt.value);
        
        if (selectedTables.length > 0) {
            additionalColumnsContainer.innerHTML = '';
            selectedTables.forEach(tableName => {
                const tableInfoDiv = document.createElement('div');
                tableInfoDiv.className = 'table-info';
                
                const tableTitle = document.createElement('h4');
                tableTitle.textContent = tableName;
                tableInfoDiv.appendChild(tableTitle);
                
                const columnsList = document.createElement('ul');
                columnsList.className = 'column-list';
                
                if (tableSchemas[tableName]) {
                    tableSchemas[tableName].forEach(column => {
                        const columnItem = document.createElement('li');
                        columnItem.textContent = column;
                        columnsList.appendChild(columnItem);
                    });
                }
                
                tableInfoDiv.appendChild(columnsList);
                additionalColumnsContainer.appendChild(tableInfoDiv);
            });
            
            additionalSchemaDisplay.classList.remove('hidden');
        } else {
            additionalSchemaDisplay.classList.add('hidden');
            additionalColumnsContainer.innerHTML = '';
        }
        
        updateGenerateButtonState();
    }

    function displayTableSchema(tableName, containerElement) {
        containerElement.innerHTML = '';
        
        if (tableSchemas[tableName]) {
            tableSchemas[tableName].forEach(column => {
                const listItem = document.createElement('li');
                listItem.textContent = column;
                containerElement.appendChild(listItem);
            });
        }
    }

    function generateSqlQuery() {
        const primaryTable = primaryTableSelect.value;
        const additionalTables = Array.from(additionalTablesSelect.selectedOptions).map(opt => opt.value);
        const naturalQuery = naturalQueryInput.value.trim();
        
        if (!primaryTable || !naturalQuery) {
            showFeedback('Please select a table and enter a query.', 'error');
            return;
        }
        
        const requestData = {
            natural_language_query: naturalQuery,
            primary_table_name: primaryTable,
            additional_tables: additionalTables
        };
        
        showLoader();
        fetch(GENERATE_SQL_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(error => {
                    throw new Error(error.detail || 'SQL generation failed');
                });
            }
            return response.json();
        })
        .then(data => {
            generatedSqlTextarea.value = data.sql_query;
            sqlSection.classList.remove('hidden');
            executeSqlButton.disabled = false;
            hideLoader();
        })
        .catch(error => {
            console.error('Error generating SQL:', error);
            showFeedback(`SQL generation failed: ${error.message}`, 'error');
            hideLoader();
        });
    }

    function executeSqlQuery() {
        const primaryTable = primaryTableSelect.value;
        const additionalTables = Array.from(additionalTablesSelect.selectedOptions).map(opt => opt.value);
        const sqlQuery = generatedSqlTextarea.value.trim();
        
        if (!sqlQuery) {
            showFeedback('No SQL query to execute.', 'error');
            return;
        }
        
        const requestData = {
            sql_query: sqlQuery,
            primary_table_name: primaryTable,
            additional_tables: additionalTables
        };
        
        showLoader();
        fetch(EXECUTE_SQL_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(error => {
                    throw new Error(error.detail || 'SQL execution failed');
                });
            }
            return response.json();
        })
        .then(data => {
            displayQueryResults(data);
            resultsSection.classList.remove('hidden');
            hideLoader();
        })
        .catch(error => {
            console.error('Error executing SQL:', error);
            showFeedback(`SQL execution failed: ${error.message}`, 'error');
            hideLoader();
        });
    }

    function displayQueryResults(data) {
        resultsTableContainer.innerHTML = '';
        
        if (data.row_count === 0) {
            resultsTableContainer.innerHTML = '<p>No results found.</p>';
            return;
        }
        
        const table = document.createElement('table');
        table.id = 'resultsTable';
        
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        data.columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = column;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        const tbody = document.createElement('tbody');
        
        data.results.forEach(row => {
            const tr = document.createElement('tr');
            
            data.columns.forEach(column => {
                const td = document.createElement('td');
                td.textContent = row[column] !== null && row[column] !== undefined ? row[column] : '';
                tr.appendChild(td);
            });
            
            tbody.appendChild(tr);
        });
        
        table.appendChild(tbody);
        resultsTableContainer.appendChild(table);
        
        const summary = document.createElement('p');
        summary.textContent = `${data.row_count} result${data.row_count !== 1 ? 's' : ''} found.`;
        resultsTableContainer.appendChild(summary);
    }

    function updateGenerateButtonState() {
        const primaryTable = primaryTableSelect.value;
        const naturalQuery = naturalQueryInput.value.trim();
        generateSqlButton.disabled = !(primaryTable && naturalQuery);
    }

    function showFeedback(message, type) {
        feedbackArea.textContent = message;
        feedbackArea.className = `feedback ${type}`;
        
        setTimeout(() => {
            feedbackArea.className = 'feedback';
        }, 5000);
    }

    function showLoader() {
        loader.classList.remove('hidden');
    }

    function hideLoader() {
        loader.classList.add('hidden');
    }
});