const SUPABASE_URL = 'https://ftysxafemyehddtnhydw.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ0eXN4YWZlbXllaGRkdG5oeWR3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjIzNjYwMTcsImV4cCI6MjA3Nzk0MjAxN30.jIs89pBcUZJpMWLpKVqxs48EfnLqIrokEb9OVz9UsxY';

const supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

let map;
let currentUser = null;

document.addEventListener('DOMContentLoaded', async () => {
    await checkAuth();
    initPrecipSlider();
});

async function checkAuth() {
    const { data: { session } } = await supabase.auth.getSession();

    if (session) {
        currentUser = session.user;
        showApp();
        initMap();
        loadStats();
    } else {
        showAuthModal();
    }

    supabase.auth.onAuthStateChange((async (event, session) => {
        if (event === 'SIGNED_IN' && session) {
            currentUser = session.user;
            showApp();
            if (!map) {
                initMap();
                loadStats();
            }
        } else if (event === 'SIGNED_OUT') {
            currentUser = null;
            showAuthModal();
            if (map) {
                map.remove();
                map = null;
            }
        }
    }));
}

function showAuthModal() {
    document.getElementById('authModal').classList.remove('hidden');
    document.getElementById('appContainer').classList.add('hidden');
}

function closeAuthModal() {
    document.getElementById('authModal').classList.add('hidden');
}

function showApp() {
    document.getElementById('authModal').classList.add('hidden');
    document.getElementById('appContainer').classList.remove('hidden');

    if (currentUser) {
        const userName = currentUser.user_metadata?.full_name || currentUser.email;
        document.getElementById('userName').textContent = userName;
    }
}

function switchToSignup() {
    document.getElementById('loginForm').classList.add('hidden');
    document.getElementById('signupForm').classList.remove('hidden');
    document.getElementById('authTitle').textContent = 'Create Account';
    clearAuthMessages();
}

function switchToLogin() {
    document.getElementById('signupForm').classList.add('hidden');
    document.getElementById('loginForm').classList.remove('hidden');
    document.getElementById('authTitle').textContent = 'Welcome to Flood Sentinel';
    clearAuthMessages();
}

function showAuthError(message) {
    const errorEl = document.getElementById('authError');
    errorEl.textContent = message;
    errorEl.classList.remove('hidden');
}

function showAuthSuccess(message) {
    const successEl = document.getElementById('authSuccess');
    successEl.textContent = message;
    successEl.classList.remove('hidden');
}

function clearAuthMessages() {
    document.getElementById('authError').classList.add('hidden');
    document.getElementById('authSuccess').classList.add('hidden');
}

async function handleLogin() {
    clearAuthMessages();

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    if (!email || !password) {
        showAuthError('Please enter email and password');
        return;
    }

    const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password
    });

    if (error) {
        showAuthError(error.message);
    } else {
        showAuthSuccess('Login successful!');
        setTimeout(() => {
            closeAuthModal();
        }, 500);
    }
}

async function handleSignup() {
    clearAuthMessages();

    const name = document.getElementById('signupName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;

    if (!email || !password) {
        showAuthError('Please enter email and password');
        return;
    }

    if (password.length < 6) {
        showAuthError('Password must be at least 6 characters');
        return;
    }

    const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
            data: {
                full_name: name
            }
        }
    });

    if (error) {
        showAuthError(error.message);
    } else {
        await supabase.from('user_profiles').insert({
            id: data.user.id,
            email: email,
            full_name: name
        });

        showAuthSuccess('Account created successfully!');
        setTimeout(() => {
            switchToLogin();
        }, 2000);
    }
}

async function handleLogout() {
    await supabase.auth.signOut();
    showAuthModal();
}

function initMap() {
    map = L.map('map').setView([29.7604, -95.3698], 11);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    loadInitialData();
}

async function loadInitialData() {
    try {
        const response = await fetch('/api/risk-predictions');
        const data = await response.json();

        if (data.features) {
            data.features.forEach(feature => {
                const coords = feature.geometry.coordinates;
                const props = feature.properties;

                const circle = L.circleMarker([coords[1], coords[0]], {
                    radius: 6,
                    fillColor: props.color,
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.7
                });

                circle.bindPopup(`
                    <strong>Risk Level: ${props.risk_level}</strong><br>
                    Probability: ${(props.flood_probability * 100).toFixed(1)}%<br>
                    Elevation: ${props.elevation_m.toFixed(1)}m
                `);

                circle.addTo(map);
            });
        }
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        document.getElementById('totalLocations').textContent = stats.total_locations.toLocaleString();
        document.getElementById('atRisk').textContent = stats.homes_at_risk.toLocaleString();
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function initPrecipSlider() {
    const slider = document.getElementById('precipSlider');
    const value = document.getElementById('precipValue');

    slider.addEventListener('input', (e) => {
        value.textContent = e.target.value + '"';
    });
}

async function updateRiskMap() {
    const precip = parseFloat(document.getElementById('precipSlider').value);
    const mode = document.getElementById('modeSelect').value;

    console.log('Updating risk map with precip:', precip, 'mode:', mode);
    alert('Risk map update feature coming soon!');
}
