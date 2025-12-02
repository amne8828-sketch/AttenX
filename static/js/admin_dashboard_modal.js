// Admin Dashboard Modal Functions

async function showRequestsModal() {
    const modal = document.getElementById('requestsModal');
    const requestsList = document.getElementById('requestsList');

    if (!modal || !requestsList) {
        console.error('Modal elements not found');
        return;
    }

    modal.style.display = 'flex';
    requestsList.innerHTML = '<div style="text-align: center; padding: 2rem;"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        const response = await fetch('/api/enrollment_requests');
        const data = await response.json();
        const requests = data.requests || data || [];

        if (requests.length === 0) {
            requestsList.innerHTML = '<div style="text-align: center; padding: 2rem; color: #6c757d;">No pending requests</div>';
            return;
        }

        let html = '<div style="display: grid; gap: 1rem;">';
        requests.forEach(req => {
            html += `
                <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; background: #f8f9fa;">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                        <div>
                            <strong>${req.name || 'Unknown'}</strong>
                            <div style="font-size: 0.875rem; color: #6c757d;">${req.email || 'N/A'}</div>
                        </div>
                        <span class="badge bg-warning text-dark">${req.status || 'pending'}</span>
                    </div>
                    <div style="font-size: 0.875rem; margin-bottom: 0.5rem;">
                        <span>📸 ${req.image_count || 0} images</span> • 
                        <span>📅 ${new Date(req.submitted_at || Date.now()).toLocaleDateString()}</span>
                    </div>
                    <div style="display: flex; gap: 0.5rem; margin-top: 0.75rem;">
                        <button class="btn btn-success btn-sm" onclick="approveRequest('${req._id}')">
                            <i class="fas fa-check"></i> Approve
                        </button>
                        <button class="btn btn-danger btn-sm" onclick="rejectRequest('${req._id}')">
                            <i class="fas fa-times"></i> Reject
                        </button>
                        <button class="btn btn-info btn-sm" onclick="viewRequestDetails('${req._id}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </div>
                </div>
            `;
        });
        html += '</div>';

        requestsList.innerHTML = html;
    } catch (error) {
        console.error('Error loading requests:', error);
        requestsList.innerHTML = '<div style="text-align: center; padding: 2rem; color: #dc3545;">Failed to load requests</div>';
    }
}

function closeRequestsModal() {
    const modal = document.getElementById('requestsModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

async function approveRequest(requestId) {
    if (!confirm('Approve this enrollment request?')) return;

    try {
        const response = await fetch(`/api/approve_enrollment/${requestId}`, {
            method: 'POST'
        });

        if (response.ok) {
            alert('Request approved successfully!');
            showRequestsModal(); // Reload the list
            if (typeof loadUsers === 'function') loadUsers(); // Refresh users list if function exists
            if (typeof loadStats === 'function') loadStats(); // Refresh stats if function exists
        } else {
            const data = await response.json();
            alert(data.msg || 'Failed to approve');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error approving request');
    }
}

async function rejectRequest(requestId) {
    const reason = prompt('Enter rejection reason:');
    if (!reason) return;

    try {
        const response = await fetch(`/api/reject_enrollment/${requestId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ reason })
        });

        if (response.ok) {
            alert('Request rejected');
            showRequestsModal(); // Reload the list
        } else {
            const data = await response.json();
            alert(data.msg || 'Failed to reject');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error rejecting request');
    }
}

function viewRequestDetails(requestId) {
    window.location.href = `/admin/enrollment-requests?request=${requestId}`;
}

// Expose functions globally
window.showRequestsModal = showRequestsModal;
window.closeRequestsModal = closeRequestsModal;
window.approveRequest = approveRequest;
window.rejectRequest = rejectRequest;
window.viewRequestDetails = viewRequestDetails;
