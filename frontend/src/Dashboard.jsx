import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Users, UserCheck, Activity, Calendar, MoreVertical, ArrowUpRight } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, AreaChart, Area } from 'recharts';

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
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return (
        <div className="flex items-center justify-center h-[80vh]">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
    );

    const emotionData = stats ? Object.entries(stats.emotion_distribution).map(([name, value]) => ({ name, value })) : [];
    const COLORS = ['#6366f1', '#ec4899', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#64748b'];

    return (
        <>
            <div className="mb-8 flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-800">Dashboard Overview</h1>
                    <p className="text-slate-500 mt-1">Welcome back, Admin</p>
                </div>
                <button className="bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2.5 rounded-xl font-medium transition-colors shadow-lg shadow-indigo-200">
                    Generate Report
                </button>
            </div>

            {/* Premium Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard
                    title="Total Present"
                    value={stats?.total_present_today || 0}
                    icon={<UserCheck className="text-white" size={24} />}
                    color="bg-gradient-to-br from-blue-500 to-blue-600"
                    trend="+12% from yesterday"
                />
                <StatCard
                    title="Unique Students"
                    value={stats?.unique_students_today || 0}
                    icon={<Users className="text-white" size={24} />}
                    color="bg-gradient-to-br from-pink-500 to-pink-600"
                    trend="Active today"
                />
                <StatCard
                    title="System Status"
                    value="Online"
                    icon={<Activity className="text-white" size={24} />}
                    color="bg-gradient-to-br from-emerald-500 to-emerald-600"
                    trend="Monitoring live"
                />
                <StatCard
                    title="Date"
                    value={new Date().toLocaleDateString('en-US', { day: 'numeric', month: 'short' })}
                    icon={<Calendar className="text-white" size={24} />}
                    color="bg-gradient-to-br from-amber-500 to-amber-600"
                    trend={new Date().getFullYear()}
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Content - Attendance Log */}
                <div className="lg:col-span-2 glass rounded-2xl p-6 card-hover">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-lg font-bold text-slate-800">Recent Attendance</h2>
                        <button className="text-slate-400 hover:text-indigo-600 transition-colors">
                            <MoreVertical size={20} />
                        </button>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="text-slate-400 text-xs uppercase tracking-wider border-b border-slate-100">
                                    <th className="pb-4 font-semibold pl-4">Student</th>
                                    <th className="pb-4 font-semibold">Status</th>
                                    <th className="pb-4 font-semibold">Time</th>
                                    <th className="pb-4 font-semibold">ID</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm">
                                {attendance.map((record) => (
                                    <tr key={record.id} className="group hover:bg-slate-50/80 transition-colors border-b border-slate-50 last:border-0">
                                        <td className="py-4 pl-4">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-sm">
                                                    {record.name.charAt(0)}
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-slate-800">{record.name}</p>
                                                    <p className="text-xs text-slate-400">Class A</p>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="py-4">
                                            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-700 border border-emerald-200">
                                                {record.status}
                                            </span>
                                        </td>
                                        <td className="py-4 text-slate-500 font-medium">
                                            {new Date(record.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </td>
                                        <td className="py-4 text-slate-400">#{record.id}</td>
                                    </tr>
                                ))}
                                {attendance.length === 0 && (
                                    <tr>
                                        <td colSpan="4" className="py-12 text-center text-slate-400">
                                            No attendance records found today.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Sidebar - Emotion Stats */}
                <div className="glass rounded-2xl p-6 card-hover flex flex-col">
                    <div className="flex items-center justify-between mb-6">
                        <h2 className="text-lg font-bold text-slate-800">Class Mood</h2>
                        <div className="bg-green-100 text-green-700 px-2 py-1 rounded text-xs font-bold">Live</div>
                    </div>

                    <div className="h-64 relative">
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
                                    stroke="none"
                                >
                                    {emotionData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                        {/* Center Text */}
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
                            <p className="text-2xl font-bold text-slate-800">{emotionData.reduce((a, b) => a + b.value, 0)}</p>
                            <p className="text-xs text-slate-400">Total</p>
                        </div>
                    </div>

                    <div className="mt-6 space-y-4 flex-1 overflow-y-auto pr-2 custom-scrollbar">
                        {emotionData.map((entry, index) => (
                            <div key={entry.name} className="flex items-center justify-between p-3 rounded-xl hover:bg-slate-50 transition-colors group">
                                <div className="flex items-center gap-3">
                                    <div className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                                    <span className="text-sm font-medium text-slate-600 capitalize group-hover:text-slate-900 transition-colors">{entry.name}</span>
                                </div>
                                <span className="font-bold text-slate-800">{entry.value}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </>
    );
};

const StatCard = ({ title, value, icon, color, trend }) => (
    <div className={`rounded-2xl p-6 text-white shadow-lg transform transition-all hover:-translate-y-1 hover:shadow-xl ${color}`}>
        <div className="flex justify-between items-start">
            <div>
                <p className="text-white/80 text-sm font-medium mb-1">{title}</p>
                <h3 className="text-3xl font-bold mb-4">{value}</h3>
                <div className="flex items-center gap-1 text-white/90 text-xs bg-white/20 w-fit px-2 py-1 rounded-lg backdrop-blur-sm">
                    <ArrowUpRight size={14} />
                    <span>{trend}</span>
                </div>
            </div>
            <div className="p-3 bg-white/20 rounded-xl backdrop-blur-sm">
                {icon}
            </div>
        </div>
    </div>
);

export default Dashboard;
