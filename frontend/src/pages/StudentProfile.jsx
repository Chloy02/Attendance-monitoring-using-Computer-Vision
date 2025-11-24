import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { ArrowLeft, Calendar, User, Clock, Smile } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const API_URL = 'http://localhost:8000';

const StudentProfile = () => {
    const { name } = useParams();
    const navigate = useNavigate();
    const [attendance, setAttendance] = useState([]);
    const [emotions, setEmotions] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [attendanceRes, emotionsRes] = await Promise.all([
                    axios.get(`${API_URL}/attendance?name=${name}&limit=50`),
                    axios.get(`${API_URL}/emotions/${name}`)
                ]);
                setAttendance(attendanceRes.data);
                setEmotions(emotionsRes.data);
                setLoading(false);
            } catch (error) {
                console.error("Error fetching profile data:", error);
                setLoading(false);
            }
        };
        fetchData();
    }, [name]);

    if (loading) return <div className="p-8 text-center">Loading Profile...</div>;

    // Process emotion data for chart
    const emotionCounts = emotions.reduce((acc, curr) => {
        acc[curr.emotion] = (acc[curr.emotion] || 0) + 1;
        return acc;
    }, {});
    const emotionData = Object.entries(emotionCounts).map(([key, value]) => ({ name: key, value }));
    const COLORS = ['#6366f1', '#ec4899', '#10b981', '#f59e0b', '#ef4444'];

    return (
        <div className="space-y-8">
            <button
                onClick={() => navigate(-1)}
                className="flex items-center gap-2 text-slate-500 hover:text-indigo-600 transition-colors"
            >
                <ArrowLeft size={20} />
                Back
            </button>

            {/* Profile Header */}
            <div className="bg-white p-8 rounded-2xl border border-slate-100 shadow-sm flex flex-col md:flex-row items-center gap-8">
                <div className="w-24 h-24 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center text-3xl font-bold">
                    {name.charAt(0)}
                </div>
                <div className="text-center md:text-left flex-1">
                    <h1 className="text-3xl font-bold text-slate-800">{name}</h1>
                    <p className="text-slate-500">Student ID: #{Math.floor(Math.random() * 1000) + 1000}</p>
                    <div className="flex flex-wrap justify-center md:justify-start gap-4 mt-4">
                        <div className="flex items-center gap-2 px-4 py-2 bg-slate-50 rounded-lg text-sm font-medium text-slate-600">
                            <Calendar size={16} />
                            {attendance.length} Days Present
                        </div>
                        <div className="flex items-center gap-2 px-4 py-2 bg-green-50 rounded-lg text-sm font-medium text-green-700">
                            <User size={16} />
                            Active Status
                        </div>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Attendance History */}
                <div className="lg:col-span-2 bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
                    <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                        <Clock size={20} className="text-indigo-600" />
                        Attendance History
                    </h2>
                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="text-slate-400 text-xs uppercase tracking-wider border-b border-slate-100">
                                    <th className="pb-3 pl-4">Date</th>
                                    <th className="pb-3">Time</th>
                                    <th className="pb-3">Status</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm">
                                {attendance.map((record) => (
                                    <tr key={record.id} className="border-b border-slate-50 hover:bg-slate-50/50 transition-colors">
                                        <td className="py-3 pl-4 font-medium text-slate-700">
                                            {new Date(record.timestamp).toLocaleDateString()}
                                        </td>
                                        <td className="py-3 text-slate-500">
                                            {new Date(record.timestamp).toLocaleTimeString()}
                                        </td>
                                        <td className="py-3">
                                            <span className="px-2 py-1 rounded-md text-xs font-medium bg-emerald-100 text-emerald-700">
                                                {record.status}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Emotion Analysis */}
                <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
                    <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                        <Smile size={20} className="text-pink-500" />
                        Emotion Trends
                    </h2>
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
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="mt-4 space-y-2">
                        {emotionData.map((entry, index) => (
                            <div key={entry.name} className="flex items-center justify-between text-sm">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                                    <span className="capitalize text-slate-600">{entry.name}</span>
                                </div>
                                <span className="font-bold text-slate-800">{entry.value}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StudentProfile;
