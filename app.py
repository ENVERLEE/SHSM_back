# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import requests
import time
import random
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import certifi
import PyPDF2
import io
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import trafilatura

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MongoDB Atlas configuration
mongo_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
db = client[os.getenv('MONGODB_DB', 'research_automation')]

# Ensure indexes
db.projects.create_index([('title', 'text')])
db.projects.create_index([('created_at', DESCENDING)])

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# App configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}

# Setup logging
if not app.debug:
    os.makedirs('logs', exist_ok=True)
    file_handler = RotatingFileHandler(
        'logs/research_automation.log',
        maxBytes=10240,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Research Automation API startup')

# Helper functions
def serialize_objectid(obj):
    """Convert ObjectId to string in dictionary"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, ObjectId):
                obj[k] = str(v)
            elif isinstance(v, (dict, list)):
                serialize_objectid(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                serialize_objectid(item)
    return obj

# app.py의 AIService 클래스에 추가

class AIService:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')

    def parse_research_plan(self, text):
        """AI 응답을 파싱하여 JSON 형식으로 변환"""
        try:
            # 먼저 직접 JSON 파싱 시도
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                pass

            # 텍스트를 단계별로 분리
            steps = []
            step_pattern = r'단계\s*\d+'
            step_texts = re.split(step_pattern, text)
            
            for step_text in step_texts[1:]:  # 첫 번째는 건너뛰기
                if not step_text.strip():
                    continue

                step_data = {
                    'description': '',
                    'keywords': '',
                    'methodology': '',
                    'output_format': ''
                }

                # 설명 추출
                desc_patterns = [
                    r'설명\s*:\s*([^키워드:방법:결과물:]+)',
                    r'목표\s*:\s*([^키워드:방법:결과물:]+)'
                ]
                for pattern in desc_patterns:
                    match = re.search(pattern, step_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    if match:
                        step_data['description'] = self.clean_text(match.group(1))
                        break

                # 키워드 추출
                keyword_match = re.search(r'키워드\s*:\s*([^방법:결과물:]+)', step_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if keyword_match:
                    step_data['keywords'] = self.clean_text(keyword_match.group(1))

                # 연구 방법 추출
                method_patterns = [
                    r'연구\s*방법\s*:\s*([^키워드:결과물:]+)',
                    r'방법\s*:\s*([^키워드:결과물:]+)',
                    r'methodology\s*:\s*([^keywords:output:]+)'
                ]
                for pattern in method_patterns:
                    match = re.search(pattern, step_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    if match:
                        step_data['methodology'] = self.clean_text(match.group(1))
                        break

                # 결과물 형식 추출
                output_patterns = [
                    r'결과물\s*형식\s*:\s*([^참고:주의:]+)',
                    r'결과물\s*:\s*([^참고:주의:]+)',
                    r'output\s*format\s*:\s*([^note:reference:]+)'
                ]
                for pattern in output_patterns:
                    match = re.search(pattern, step_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    if match:
                        step_data['output_format'] = self.clean_text(match.group(1))
                        break

                # 누락된 필드 재시도
                if not step_data['methodology']:
                    method_match = re.search(r'(?:연구\s*방법|방법론|방법)\s*:\s*([^\n]+)', 
                                          step_text, re.IGNORECASE | re.MULTILINE)
                    if method_match:
                        step_data['methodology'] = self.clean_text(method_match.group(1))

                # 최소한 하나의 필드가 있는 경우만 추가
                if any(step_data.values()):
                    steps.append(step_data)

            return steps

        except Exception as e:
            app.logger.error(f"연구 계획 파싱 중 오류: {str(e)}")
            return None

    def clean_text(self, text):
        """텍스트 정제"""
        if not text:
            return ''
        
        # 불필요한 공백 및 특수문자 제거
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'["\'\[\]{}]', '', text)
        
        # 중복된 구분자 제거
        text = re.sub(r'[:：\-=>\)]\s*[:：\-=>\)]', ':', text)
        
        # 앞뒤 특수문자 및 공백 제거
        text = text.strip('.:：-=> \n\t')
        
        # 참고 및 주의사항 섹션 제거
        patterns = [
            r'참고\s*:[^*#\n]+',
            r'주의\s*:[^*#\n]+',
            r'Note\s*:[^*#\n]+',
            r'Reference\s*:[^*#\n]+'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            
        return text.strip()

    def generate_research_plan(self, evaluation_plan, submission_format):
        """연구 계획 생성 및 파싱"""
        try:
            prompt = (
                "너는 이제부터 한국의 안보 전문 연구자가 되어 특정 연구역량 평가를 보게될거야. "
                "너가 이 평가를 통과하지 않으면 큰 불이익을 받게돼. "
                "평가 계획안과 제출양식을 보내줄테니, 연구 과정 단계를 세세하게 3-6개 정도 순서대로 짜주고 "
                "그 과정마다 필요한 참고자료를 찾을때 필요한 keyword와 연구 방법을 같이 제시해주세요.\n\n"
                f"평가계획서:\n{evaluation_plan}\n\n"
                f"제출양식:\n{submission_format}\n\n"
                "다음 형식으로 응답해주세요:\n"
                "단계 1:\n"
                "설명: [설명]\n"
                "키워드: [키워드들]\n"
                "방법: [research method]\n"
                "결과물 형식: [output format]\n\n"
                "단계 2: ..."
            )

            response = self._call_groq_api(prompt)
            if not response:
                return None

            # AI 응답을 파싱
            research_steps = self.parse_research_plan(response)
            if not research_steps:
                app.logger.error("연구 계획 파싱 실패")
                return None

            return research_steps

        except Exception as e:
            app.logger.error(f"연구 계획 생성 중 오류: {str(e)}")
            return None


    def search_papers(self, keywords, num_results=5):
        try:
            # Google Custom Search API 호출
            search_results = self._search_google_scholar(keywords, num_results)
            processed_results = []
            
            for result in search_results:
                processed_result = self._process_search_result(result)
                if processed_result:
                    processed_results.append(processed_result)
                    time.sleep(random.uniform(1, 2))  # Rate limiting
                    
            return processed_results
        except Exception as e:
            app.logger.error(f"Paper search error: {str(e)}")
            return []

    def execute_research_step(self, step_description, methodology, output_format, references):
        try:
            context = "참고문헌:\n" + "\n".join(references) if references else ""
            
            prompt = (
                "다음 연구 단계를 수행해주세요. 제시된 참고문헌을 활용하여 결과를 도출하세요.\n\n"
                f"단계 설명: {step_description}\n"
                f"연구 방법: {methodology}\n"
                f"결과물 양식: {output_format}\n\n"
                f"{context}"
            )

            return self._call_groq_api(prompt)
        except Exception as e:
            app.logger.error(f"Research step execution error: {str(e)}")
            return None

    def generate_final_report(self, submission_format, research_results):
        try:
            prompt = (
                "다음 연구 결과들을 제출 양식에 맞게 최종 보고서로 가공해주세요.\n\n"
                f"제출 양식:\n{submission_format}\n\n"
                f"연구 결과:\n{json.dumps(research_results, ensure_ascii=False, indent=2)}"
            )

            return self._call_groq_api(prompt)
        except Exception as e:
            app.logger.error(f"Final report generation error: {str(e)}")
            return None

    def _call_groq_api(self, prompt):
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gemma2-9b-it",
                "messages": [
                    {"role": "system", "content": "You are a research assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 4096
            }

            response = requests.post(
                self.groq_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return None

        except Exception as e:
            app.logger.error(f"Groq API call error: {str(e)}")
            return None

    def _search_google_scholar(self, keywords, num_results):
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            query = f"scholarly articles research papers {' '.join(keywords)}"
            
            params = {
                'key': self.google_api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),
                'sort': 'date',
                'gl': 'kr',
                'lr': 'lang_ko|lang_en'
            }

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
            return []

        except Exception as e:
            app.logger.error(f"Google Scholar search error: {str(e)}")
            return []

    def _process_search_result(self, item):
        try:
            url = item.get('link', '')
            if not self._is_valid_url(url):
                return None

            # Extract text from webpage
            text = self._extract_webpage_text(url)
            if not text:
                return None

            # Summarize content
            summary = self._summarize_content(text)
            if not summary:
                return None

            return {
                'title': item.get('title', '')[:255],
                'content': summary,
                'url': url,
                'metavalue': {
                    'source': urlparse(url).netloc,
                    'summary_length': len(summary)
                }
            }

        except Exception as e:
            app.logger.warning(f"Search result processing error: {str(e)}")
            return None

    def _extract_webpage_text(self, url):
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded)
                if text:
                    return text

            response = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=30
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Clean up HTML
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
                
            return ' '.join(soup.stripped_strings)

        except Exception as e:
            app.logger.error(f"Webpage text extraction error: {str(e)}")
            return None

    def _summarize_content(self, text):
        try:
            prompt = (
                "아래 텍스트를 읽고 주요 내용을 500자 이내로 요약해주세요. "
                "학술적인 내용이므로 전문적이고 객관적인 톤을 유지해주세요. "
                "중요한 연구 결과, 방법론, 결론 위주로 요약해주세요.\n\n"
                f"텍스트: {text[:4000]}"
            )
            
            return self._call_groq_api(prompt)

        except Exception as e:
            app.logger.error(f"Content summarization error: {str(e)}")
            return None

    def _is_valid_url(self, url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

# Initialize AI Service
ai_service = AIService()

# API Routes
@app.route('/api/process-cdn-file', methods=['POST'])
def process_cdn_file():
    try:
        data = request.json
        if not data or 'file_url' not in data:
            return jsonify({
                'success': False,
                'message': 'File URL is required'
            }), 400

        file_url = data['file_url']
        file_type = data.get('file_type')

        response = requests.get(file_url)
        if response.status_code != 200:
            return jsonify({
                'success': False,
                'message': 'Failed to download file from CDN'
            }), 400

        content_type = response.headers.get('content-type', '')
        extracted_text = ''

        if 'pdf' in content_type:
            try:
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"

                if not extracted_text.strip():
                    images = convert_from_bytes(response.content)
                    for image in images:
                        text = pytesseract.image_to_string(image, lang='kor+eng')
                        extracted_text += text + "\n"

            except Exception as e:
                app.logger.error(f"PDF processing error: {str(e)}")

        elif 'image' in content_type:
            try:
                image = Image.open(io.BytesIO(response.content))
                if image.mode != 'L':
                    image = image.convert('L')
                extracted_text = pytesseract.image_to_string(image, lang='kor+eng')
            except Exception as e:
                app.logger.error(f"Image processing error: {str(e)}")

        if not extracted_text.strip():
            return jsonify({
                'success': False,
                'message': 'Failed to extract text from file'
            }), 400

        return jsonify({
            'success': True,
            'text': extracted_text.strip(),
            'file_type': file_type
        })

    except Exception as e:
        app.logger.error(f"File processing error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error processing file: {str(e)}'
        }), 500

@app.route('/api/project', methods=['POST'])
def create_project():
    try:
        # 1. Validate request data
        data = request.json
        required_fields = ['title', 'evaluation_plan', 'submission_format']
        
        if not data or not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'message': 'Missing required fields'
            }), 400

        # 2. Generate research plan with integrated parsing
        research_steps = ai_service.generate_research_plan(
            data['evaluation_plan'],
            data['submission_format']
        )

        if not research_steps:
            return jsonify({
                'success': False,
                'message': 'Failed to generate research plan'
            }), 500

        # 3. Create project document
        project = {
            'title': data['title'],
            'evaluation_plan': data['evaluation_plan'],
            'evaluation_plan_file': data.get('evaluation_plan_file'),
            'submission_format': data['submission_format'],
            'submission_format_file': data.get('submission_format_file'),
            'created_at': datetime.utcnow(),
            'research_steps': [],  # Will be populated step by step
            'references': [],      # Will be populated with references
            'status': 'in_progress',
            'last_updated': datetime.utcnow()
        }
        
        # 4. Insert initial project document
        result = db.projects.insert_one(project)
        project_id = result.inserted_id

        # 5. Process each research step and generate references
        processed_steps = []
        
        for i, step in enumerate(research_steps, 1):
            try:
                # Add step number and timestamps
                step_data = {
                    'step_number': i,
                    'description': step.get('description', '').strip(),
                    'keywords': step.get('keywords', '').strip(),
                    'methodology': step.get('methodology', '').strip(),
                    'output_format': step.get('output_format', '').strip(),
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow(),
                    'status': 'pending',
                    'result': None
                }
                
                # Generate references for the step
                keywords = [kw.strip() for kw in step_data['keywords'].split(',') if kw.strip()]
                step_references = []
                
                if keywords:
                    references = ai_service.search_papers(keywords, num_results=5)
                    for ref in references:
                        if not all(key in ref for key in ['title', 'content', 'url']):
                            continue
                            
                        reference_data = {
                            'title': ref['title'][:255],
                            'content': ref['content'],
                            'url': ref['url'],
                            'metavalue': ref.get('metavalue', {}),
                            'created_at': datetime.utcnow(),
                            'step_number': i,  # Associate reference with step
                            'relevance_score': ref.get('relevance_score', 0)
                        }
                        step_references.append(reference_data)

                # Add references to step data
                step_data['references'] = step_references
                processed_steps.append(step_data)

                # Update project document with each step and its references
                db.projects.update_one(
                    {'_id': project_id},
                    {
                        '$push': {
                            'research_steps': step_data,
                            'references': {
                                '$each': step_references
                            }
                        },
                        '$set': {
                            'last_updated': datetime.utcnow()
                        }
                    }
                )

                app.logger.info(f"Processed step {i} with {len(step_references)} references")

            except Exception as step_error:
                app.logger.warning(f"Error processing step {i}: {str(step_error)}")
                # Continue with next step instead of failing completely
                continue

        # 6. Final update to project
        db.projects.update_one(
            {'_id': project_id},
            {
                '$set': {
                    'total_steps': len(processed_steps),
                    'completed_steps': 0,
                    'last_updated': datetime.utcnow()
                }
            }
        )

        # 7. Return success response with enhanced data
        return jsonify({
            'success': True,
            'project_id': str(project_id),
            'research_steps': processed_steps,
            'total_references': sum(len(step.get('references', [])) for step in processed_steps),
            'message': 'Project created successfully with research steps and references'
        })

    except Exception as e:
        app.logger.error(f"Project creation error: {str(e)}")
        # Cleanup if project was partially created
        if 'project_id' in locals():
            try:
                db.projects.delete_one({'_id': project_id})
            except Exception as cleanup_error:
                app.logger.error(f"Cleanup error: {str(cleanup_error)}")
        
        return jsonify({
            'success': False,
            'message': f'Error creating project: {str(e)}'
        }), 500


@app.route('/api/research/<project_id>', methods=['GET'])
def get_project(project_id):
    try:
        project = db.projects.find_one({'_id': ObjectId(project_id)})
        if not project:
            return jsonify({
                'success': False,
                'message': 'Project not found'
            }), 404

        return jsonify({
            'success': True,
            'project': serialize_objectid(project)
        })

    except Exception as e:
        app.logger.error(f"Error retrieving project: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/research/<project_id>/step/<int:step_number>', methods=['POST'])
def execute_research_step(project_id, step_number):
    try:
        project = db.projects.find_one({'_id': ObjectId(project_id)})
        if not project:
            return jsonify({
                'success': False,
                'message': 'Project not found'
            }), 404

        # Find the specific step
        step = next(
            (step for step in project['research_steps'] 
             if step['step_number'] == step_number),
            None
        )
        
        if not step:
            return jsonify({
                'success': False,
                'message': 'Step not found'
            }), 404

        # Get references
        references = project.get('references', [])
        reference_texts = [
            f"Title: {ref['title']}\nContent: {ref['content']}"
            for ref in references
        ]

        # Execute step
        result = ai_service.execute_research_step(
            step['description'],
            step['methodology'],
            step['output_format'],
            reference_texts
        )

        if not result:
            return jsonify({
                'success': False,
                'message': 'Failed to execute research step'
            }), 500

        # Update step result
        db.projects.update_one(
            {
                '_id': ObjectId(project_id),
                'research_steps.step_number': step_number
            },
            {'$set': {'research_steps.$.result': result}}
        )

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        app.logger.error(f"Step execution error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/research/<project_id>/finalize', methods=['POST'])
def finalize_research(project_id):
    try:
        project = db.projects.find_one({'_id': ObjectId(project_id)})
        if not project:
            return jsonify({
                'success': False,
                'message': 'Project not found'
            }), 404

        # Check if all steps are completed
        steps = project['research_steps']
        if not all(step.get('result') for step in steps):
            return jsonify({
                'success': False,
                'message': 'Not all research steps are completed'
            }), 400

        # Generate final report
        final_report = ai_service.generate_final_report(
            project['submission_format'],
            steps
        )

        if not final_report:
            return jsonify({
                'success': False,
                'message': 'Failed to generate final report'
            }), 500

        # Update project with final report
        db.projects.update_one(
            {'_id': ObjectId(project_id)},
            {'$set': {'final_report': final_report}}
        )

        return jsonify({
            'success': True,
            'final_report': final_report
        })

    except Exception as e:
        app.logger.error(f"Finalization error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    try:
        # Get total projects count
        total_projects = db.projects.count_documents({})
        
        # Get completed projects count
        completed_projects = db.projects.count_documents({
            'final_report': {'$exists': True}
        })
        
        # Get monthly project counts
        pipeline = [
            {
                '$group': {
                    '_id': {
                        'year': {'$year': '$created_at'},
                        'month': {'$month': '$created_at'}
                    },
                    'count': {'$sum': 1}
                }
            },
            {'$sort': {'_id.year': 1, '_id.month': 1}}
        ]
        
        monthly_stats = list(db.projects.aggregate(pipeline))
        
        return jsonify({
            'success': True,
            'stats': {
                'total_projects': total_projects,
                'completed_projects': completed_projects,
                'completion_rate': (completed_projects / total_projects * 100) if total_projects > 0 else 0,
                'monthly_projects': [
                    {
                        'month': f"{stat['_id']['year']}-{stat['_id']['month']:02d}",
                        'count': stat['count']
                    }
                    for stat in monthly_stats
                ]
            }
        })

    except Exception as e:
        app.logger.error(f"Dashboard stats error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/dashboard/projects', methods=['GET'])
def list_projects():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        search = request.args.get('search', '')
        
        # Build query
        query = {}
        if search:
            query['$text'] = {'$search': search}
            
        # Execute query with pagination
        total = db.projects.count_documents(query)
        projects = list(db.projects
                       .find(query)
                       .sort('created_at', -1)
                       .skip((page - 1) * per_page)
                       .limit(per_page))
        
        return jsonify({
            'success': True,
            'projects': serialize_objectid(projects),
            'total': total,
            'page': page,
            'total_pages': (total + per_page - 1) // per_page
        })

    except Exception as e:
        app.logger.error(f"Project listing error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))