import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { Search, Filter, MoreHorizontal } from 'lucide-react';

const API_URL = 'http://localhost:8000';

const Students = () => {
    const [students, setStudents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');

    useEffect(() => {
        const fetchStudents = async () => {
            try {
                const res = await axios.get(`${API_URL}/students`);
                setStudents(res.data);
                setLoading(false);
            } catch (error) {
                console.error("Error fetching students:", error);
                setLoading(false);
            }
        };
        fetchStudents();
    }, []);

    const filteredStudents = students.filter(student =>
        student.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    if (loading) return <div className="p-8 text-center">Loading Students...</div>;

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-slate-800">Students Directory</h1>
                    <p className="text-slate-500">Manage and view student records</p>
                </div>
                <div className="flex gap-3">
                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
                        <input
                            type="text"
                            placeholder="Search students..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="pl-10 pr-4 py-2 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        />
                    </div>
                    <button className="p-2 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 text-slate-600">
                        <Filter size={20} />
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredStudents.map((student, index) => (
                    <div key={index} className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center gap-4">
                                <div className="w-12 h-12 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center font-bold text-lg">
                                    {student.name.charAt(0)}
                                </div>
                                <div>
                                    <h3 className="font-bold text-slate-800">{student.name}</h3>
                                    <p className="text-xs text-slate-500">ID: {index + 1001}</p>
                                </div>
                            </div>
                            <button className="text-slate-400 hover:text-slate-600">
                                <MoreHorizontal size={20} />
                            </button>
                        </div>

                        <div className="grid grid-cols-2 gap-4 py-4 border-t border-slate-50">
                            <div>
                                <p className="text-xs text-slate-400 uppercase font-semibold">Attendance</p>
                                <p className="text-lg font-bold text-slate-800">{student.attendance_count} Days</p>
                            </div>
                            <div>
                                <p className="text-xs text-slate-400 uppercase font-semibold">Status</p>
                                <span className="inline-block px-2 py-1 mt-1 rounded-md text-xs font-medium bg-green-100 text-green-700">
                                    {student.status}
                                </span>
                            </div>
                        </div>

                        <div className="pt-4 border-t border-slate-50 flex justify-between items-center">
                            <p className="text-xs text-slate-400">Last seen: {student.last_seen ? new Date(student.last_seen).toLocaleDateString() : 'Never'}</p>
                            <Link to={`/students/${student.name}`} className="text-sm font-medium text-indigo-600 hover:text-indigo-700">View Profile</Link>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Students;
