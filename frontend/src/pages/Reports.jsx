import React, { useState } from 'react';
import { Download, Calendar, FileText } from 'lucide-react';

const API_URL = 'http://localhost:8000';

const Reports = () => {
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [downloading, setDownloading] = useState(false);

    const handleDownload = async () => {
        setDownloading(true);
        try {
            const queryParams = new URLSearchParams();
            if (startDate) queryParams.append('start_date', startDate);
            if (endDate) queryParams.append('end_date', endDate);

            const response = await fetch(`${API_URL}/reports/export?${queryParams.toString()}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `attendance_report_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error("Download failed:", error);
        } finally {
            setDownloading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <div>
                <h1 className="text-2xl font-bold text-slate-800">Reports & Analytics</h1>
                <p className="text-slate-500">Generate and download attendance records</p>
            </div>

            <div className="bg-white p-8 rounded-2xl border border-slate-100 shadow-sm">
                <div className="flex items-center gap-4 mb-8">
                    <div className="p-3 bg-indigo-50 rounded-xl text-indigo-600">
                        <FileText size={24} />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-slate-800">Attendance Report</h2>
                        <p className="text-sm text-slate-500">Export detailed logs as CSV</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-700">Start Date</label>
                        <div className="relative">
                            <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                            <input
                                type="date"
                                value={startDate}
                                onChange={(e) => setStartDate(e.target.value)}
                                className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            />
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-slate-700">End Date</label>
                        <div className="relative">
                            <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                                className="w-full pl-10 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            />
                        </div>
                    </div>
                </div>

                <button
                    onClick={handleDownload}
                    disabled={downloading}
                    className="w-full md:w-auto flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {downloading ? (
                        <>Downloading...</>
                    ) : (
                        <>
                            <Download size={20} />
                            Download CSV Report
                        </>
                    )}
                </button>
            </div>
        </div>
    );
};

export default Reports;
