import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import papaparse from 'https://jslib.k6.io/papaparse/5.1.1/index.js';
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.1/index.js";
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";

const HOST = __ENV.SERVER_HOST || 'localhost';
const PORT = __ENV.SERVER_PORT || '8000';

const csvData = new SharedArray('test_data', function () {
    return papaparse.parse(open('./data/data_test.csv'), { header: true }).data;
});

export const options = {
    scenarios: {
        baseline_test: {
            executor: 'constant-arrival-rate',
            rate: 300,
            duration: '3m',
            preAllocatedVUs: 300,
        },
        scaling_stress_test: {
            executor: 'ramping-arrival-rate',
            startRate: 400,
            startTime: '3m',      
            timeUnit: '1s',
            stages: [
                { target: 100, duration: '1m' },
                { target: 200, duration: '1m' },
                { target: 400, duration: '1m' },
                { target: 0, duration: '30s' },
            ],
            preAllocatedVUs: 300,
            maxVUs: 500,
        },
    },
    thresholds: {
        http_req_failed: ['rate<0.01'],
        http_req_duration: ['p(99)<2000'],
        'http_req_duration{status:200}': ['p(50)<100'],
    },
};

export default function () {
    const randomItem = csvData[Math.floor(Math.random() * csvData.length)];

    let text = randomItem && randomItem.instruction ? randomItem.instruction : "";
    
    text = String(text).trim();

    if (!text) {
        console.warn(`[VU ${__VU}] WARNING: EMPTY INSTRUCTION, SKIP THIS REQUEST`);
        return;
    }
    
    const expectedLabel = randomItem.label.trim(); 

    const url = `http://${HOST}:${PORT}/v1/query-classify`;
    const payload = JSON.stringify({ text: text });
    const params = {
        headers: { 'Content-Type': 'application/json' },
    };

    const res = http.post(url, payload, params);

    check(res, {
        'is status 200': (r) => r.status === 200,
        'classification accuracy': (r) => {
            const body = r.json();
            return String(body.label).trim() === expectedLabel;
        },
    });
}

export function handleSummary(data) {
    return {
            "summary_report.html": htmlReport(data),
            stdout: textSummary(data, { indent: " ", enableColors: true }),
    };
}