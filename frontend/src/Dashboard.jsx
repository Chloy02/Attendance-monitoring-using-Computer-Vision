import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Users, UserCheck, Activity, Calendar } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const API_URL = 'http://localhost:8000';

const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [attendance, setAttendance] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const statsRes = await axios.get(`${API_URL}/stats`);
                const attendanceRes = await axios.get(`${API_URL}/attendance?limit=10`);
                setStats(statsRes.data);
                setAttendance(attendanceRes.data);
                setLoading(false);
            } catch (error) {
                console.error("Error fetching data:", error);
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 5000); // Poll every 5 seconds
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div className="flex items-center justify-center h-screen text-xl font-semibold text-slate-600">Loading Dashboard...</div>;

    // Prepare chart data
    const emotionData = stats ? Object.entries(stats.emotion_distribution).map(([name, value]) => ({ name, value })) : [];
    const COLORS = ['#6366f1', '#ec4899', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#64748b'];

    return (
        <div className="min-h-screen bg-slate-50 p-8">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-slate-900">Attendance Dashboard</h1>
                <p className="text-slate-500">Real-time monitoring system</p>
            </header>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard
                    icon={<UserCheck className="w-8 h-8 text-indigo-500" />}
                    title="Present Today"
                    value={stats?.total_present_today || 0}
                    subtitle="Total detections"
                />
                <StatCard
                    icon={<Users className="w-8 h-8 text-pink-500" />}
                    title="Unique Students"
                    value={stats?.unique_students_today || 0}
                    subtitle="Identified individuals"
                />
                <StatCard
                    icon={<Activity className="w-8 h-8 text-emerald-500" />}
                    title="System Status"
                    value="Active"
                    subtitle="Monitoring live feed"
                />
                <StatCard
                    icon={<Calendar className="w-8 h-8 text-amber-500" />}
                    title="Date"
                    value={stats?.date || '-'}
                    subtitle="Today's session"
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Content - Attendance Log */}
                <div className="lg:col-span-2 glass rounded-2xl p-6">
                    <h2 className="text-xl font-bold text-slate-800 mb-4">Recent Activity</h2>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="border-b border-slate-200 text-slate-500 text-sm">
                                    <th className="pb-3 font-medium">Time</th>
                                    <th className="pb-3 font-medium">Student Name</th>
                                    <th className="pb-3 font-medium">Status</th>
                                    <th className="pb-3 font-medium">ID</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm">
                                {attendance.map((record) => (
                                    <tr key={record.id} className="border-b border-slate-100 hover:bg-slate-50/50 transition-colors">
                                        <td className="py-3 text-slate-500">{new Date(record.timestamp).toLocaleTimeString()}</td>
                                        <td className="py-3 font-medium text-slate-900">{record.name}</td>
                                        <td className="py-3">
                                            <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700">
                                                {record.status}
                                            </span>
                                        </td>
                                        <td className="py-3 text-slate-400">#{record.id}</td>
                                    </tr>
                                ))}
                                {attendance.length === 0 && (
                                    <tr>
                                        <td colSpan="4" className="py-8 text-center text-slate-400">No attendance records found today.</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Sidebar - Emotion Stats */}
                <div className="glass rounded-2xl p-6">
                    <h2 className="text-xl font-bold text-slate-800 mb-4">Class Mood</h2>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={emotionData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {emotionData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                                <Legend />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="mt-4 space-y-3">
                        {emotionData.map((entry, index) => (
                            <div key={entry.name} className="flex items-center justify-between text-sm">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                                    <span className="text-slate-600 capitalize">{entry.name}</span>
                                </div>
                                <span className="font-semibold text-slate-900">{entry.value}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

const StatCard = ({ icon, title, value, subtitle }) => (
    <div className="glass rounded-xl p-6 flex items-start justify-between hover:shadow-lg transition-shadow">
        <div>
            <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
            <h3 className="text-2xl font-bold text-slate-900 mb-1">{value}</h3>
            <p className="text-xs text-slate-400">{subtitle}</p>
        </div>
        <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
            {icon}
        </div>
    </div>
);

export default Dashboard;
