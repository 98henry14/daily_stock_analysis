# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd
import yfinance as yf

from src.config import get_config
from src.search_service import SearchService

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    north_flow: float = 0.0             # 北向资金净流入（亿元）

    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块

    # ========== 牛市逃顶指标 ==========
    margin_balance: float = 0.0         # 两市融资余额（亿元）
    total_market_cap: float = 0.0       # 两市总市值（亿元）
    margin_ratio: float = 0.0           # 融资余额/总市值比值（%）
    is_bull_top_warning: bool = False   # 是否触发逃顶警告（比值 > 3.5%）
    margin_data_date: str = ""          # 融资数据日期


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    # 主要指数代码
    MAIN_INDICES = {
        'sh000001': '上证指数',
        'sz399001': '深证成指',
        'sz399006': '创业板指',
        'sh000688': '科创50',
        'sh000016': '上证50',
        'sh000300': '沪深300',
    }
    
    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        """
        初始化大盘分析器
        
        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        
    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据

        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)

        # 1. 获取主要指数行情
        overview.indices = self._get_main_indices()

        # 2. 获取涨跌统计
        self._get_market_statistics(overview)

        # 3. 获取板块涨跌榜
        self._get_sector_rankings(overview)

        # 4. 获取北向资金（可选）
        # self._get_north_flow(overview)

        # 5. 获取牛市逃顶指标
        self._get_bull_market_indicator(overview)

        return overview

    def _call_akshare_with_retry(self, fn, name: str, attempts: int = 2):
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(f"[大盘] {name} 获取失败 (attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 5))
        logger.error(f"[大盘] {name} 最终失败: {last_error}")
        return None
    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []
        
        try:
            logger.info("[大盘] 获取主要指数实时行情...")
            
            # 使用 akshare 获取指数行情（新浪财经接口，包含深市指数）
            df = self._call_akshare_with_retry(ak.stock_zh_index_spot_sina, "指数行情", attempts=2)
            
            if df is not None and not df.empty:
                for code, name in self.MAIN_INDICES.items():
                    # 查找对应指数
                    row = df[df['代码'] == code]
                    if row.empty:
                        # 尝试带前缀查找
                        row = df[df['代码'].str.contains(code)]
                    
                    if not row.empty:
                        row = row.iloc[0]
                        index = MarketIndex(
                            code=code,
                            name=name,
                            current=float(row.get('最新价', 0) or 0),
                            change=float(row.get('涨跌额', 0) or 0),
                            change_pct=float(row.get('涨跌幅', 0) or 0),
                            open=float(row.get('今开', 0) or 0),
                            high=float(row.get('最高', 0) or 0),
                            low=float(row.get('最低', 0) or 0),
                            prev_close=float(row.get('昨收', 0) or 0),
                            volume=float(row.get('成交量', 0) or 0),
                            amount=float(row.get('成交额', 0) or 0),
                        )
                        # 计算振幅
                        if index.prev_close > 0:
                            index.amplitude = (index.high - index.low) / index.prev_close * 100
                        indices.append(index)

            # 如果 akshare 获取失败或为空，尝试使用 yfinance 兜底
            if not indices:
                logger.warning("[大盘] 国内源获取失败，尝试使用 Yfinance 兜底...")
                indices = self._get_indices_from_yfinance()

            logger.info(f"[大盘] 获取到 {len(indices)} 个指数行情")

        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")
            # 异常时也尝试兜底
            if not indices:
                indices = self._get_indices_from_yfinance()

        return indices

    def _get_indices_from_yfinance(self) -> List[MarketIndex]:
        """从 Yahoo Finance 获取指数行情（兜底方案）"""
        indices = []
        # 映射关系：akshare代码 -> yfinance代码
        yf_mapping = {
            'sh000001': ('000001.SS', '上证指数'),
            'sz399001': ('399001.SZ', '深证成指'),
            'sz399006': ('399006.SZ', '创业板指'),
            'sh000688': ('000688.SS', '科创50'),
            'sh000016': ('000016.SS', '上证50'),
            'sh000300': ('000300.SS', '沪深300'),
        }

        try:
            for ak_code, (yf_code, name) in yf_mapping.items():
                if ak_code not in self.MAIN_INDICES:
                    continue

                ticker = yf.Ticker(yf_code)
                try:
                    hist = ticker.history(period='2d')
                    if hist.empty:
                        continue

                    today = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else today

                    price = float(today['Close'])
                    prev_close = float(prev['Close'])
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0

                    index = MarketIndex(
                        code=ak_code,
                        name=name,
                        current=price,
                        change=change,
                        change_pct=change_pct,
                        open=float(today['Open']),
                        high=float(today['High']),
                        low=float(today['Low']),
                        prev_close=prev_close,
                        volume=float(today['Volume']),
                        amount=0.0
                    )
                    indices.append(index)
                    logger.info(f"[大盘] Yfinance 成功获取: {name}")
                except Exception as e:
                    logger.debug(f"[大盘] Yfinance 获取 {name} 失败: {e}")

        except Exception as e:
            logger.error(f"[大盘] Yfinance 兜底失败: {e}")

        return indices
    
    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")
            
            # 获取全部A股实时行情
            df = self._call_akshare_with_retry(ak.stock_zh_a_spot_em, "A股实时行情", attempts=2)
            
            if df is not None and not df.empty:
                # 涨跌统计
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    overview.up_count = len(df[df[change_col] > 0])
                    overview.down_count = len(df[df[change_col] < 0])
                    overview.flat_count = len(df[df[change_col] == 0])
                    
                    # 涨停跌停统计（涨跌幅 >= 9.9% 或 <= -9.9%）
                    overview.limit_up_count = len(df[df[change_col] >= 9.9])
                    overview.limit_down_count = len(df[df[change_col] <= -9.9])
                
                # 两市成交额
                amount_col = '成交额'
                if amount_col in df.columns:
                    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                    overview.total_amount = df[amount_col].sum() / 1e8  # 转为亿元
                
                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿")
                
        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")
    
    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] 获取板块涨跌榜...")
            
            # 获取行业板块行情
            df = self._call_akshare_with_retry(ak.stock_board_industry_name_em, "行业板块行情", attempts=2)
            
            if df is not None and not df.empty:
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    df = df.dropna(subset=[change_col])
                    
                    # 涨幅前5
                    top = df.nlargest(5, change_col)
                    overview.top_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5
                    bottom = df.nsmallest(5, change_col)
                    overview.bottom_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                    logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
                    
        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
            
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
                    
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
                
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")

    def _get_bull_market_indicator(self, overview: MarketOverview):
        """
        获取牛市逃顶指标

        计算公式：融资余额 / 两市总市值
        警告阈值：> 3.5%

        数据来源：
        - 融资余额：ak.stock_margin_account_info() 获取两融账户信息
        - 总市值：ak.stock_sse_summary() + ak.stock_szse_summary()
        """
        try:
            logger.info("[大盘] 获取牛市逃顶指标...")

            # 1. 获取融资余额（两市合计）
            margin_balance = self._get_margin_balance()

            # 2. 获取两市总市值
            total_market_cap = self._get_total_market_cap()

            if margin_balance > 0 and total_market_cap > 0:
                # 计算融资余额占总市值比例
                margin_ratio = (margin_balance / total_market_cap) * 100
                overview.margin_balance = margin_balance
                overview.total_market_cap = total_market_cap
                overview.margin_ratio = round(margin_ratio, 2)
                overview.is_bull_top_warning = margin_ratio > 3.5

                warning_text = "⚠️ 触发逃顶警告!" if overview.is_bull_top_warning else "正常"
                logger.info(f"[大盘] 逃顶指标: 融资余额={margin_balance:.0f}亿, "
                          f"总市值={total_market_cap:.0f}亿, "
                          f"比值={margin_ratio:.2f}% {warning_text}")
            else:
                logger.warning(f"[大盘] 逃顶指标数据不完整: 融资余额={margin_balance}, 总市值={total_market_cap}")

        except Exception as e:
            logger.error(f"[大盘] 获取逃顶指标失败: {e}")

    def _get_margin_balance(self) -> float:
        """
        获取两市融资余额（亿元）

        数据来源：ak.stock_margin_account_info()
        返回最新一天的融资余额数据
        """
        try:
            # 获取两融账户信息
            df = self._call_akshare_with_retry(ak.stock_margin_account_info, "融资融券账户信息", attempts=2)

            if df is not None and not df.empty:
                # 按日期排序，取最新一条
                if '日期' in df.columns:
                    df['日期'] = pd.to_datetime(df['日期'])
                    df = df.sort_values('日期', ascending=False)

                latest = df.iloc[0]

                # 融资余额字段（单位：亿元）
                if '融资余额' in df.columns:
                    margin_balance = float(latest['融资余额'])
                    logger.debug(f"[大盘] 融资余额: {margin_balance:.2f}亿 (日期: {latest.get('日期', 'N/A')})")
                    return margin_balance

            return 0.0

        except Exception as e:
            logger.warning(f"[大盘] 获取融资余额失败: {e}")
            return 0.0

    def _get_total_market_cap(self) -> float:
        """
        获取两市总市值（亿元）

        计算方式：上交所总市值 + 深交所总市值
        数据来源：
        - 上交所：ak.stock_sse_summary()
        - 深交所：ak.stock_szse_summary()
        """
        total_cap = 0.0

        try:
            # 1. 获取上交所总市值
            sse_cap = self._get_sse_market_cap()
            total_cap += sse_cap

            # 2. 获取深交所总市值
            szse_cap = self._get_szse_market_cap()
            total_cap += szse_cap

            logger.debug(f"[大盘] 两市总市值: 上交所={sse_cap:.0f}亿 + 深交所={szse_cap:.0f}亿 = {total_cap:.0f}亿")

        except Exception as e:
            logger.warning(f"[大盘] 获取总市值失败: {e}")

        return total_cap

    def _get_sse_market_cap(self) -> float:
        """获取上交所总市值（亿元）"""
        try:
            df = self._call_akshare_with_retry(ak.stock_sse_summary, "上交所市值统计", attempts=2)

            if df is not None and not df.empty:
                # 查找总市值行
                cap_row = df[df['项目'] == '总市值']
                if not cap_row.empty:
                    # 单位：亿元
                    cap = float(cap_row.iloc[0].get('股票', 0) or 0)
                    return cap

                # 备选：尝试其他列名
                for col in df.columns:
                    if '市值' in str(col) or 'market' in str(col).lower():
                        return float(df[col].sum()) if df[col].dtype in ['float64', 'int64'] else 0.0

            return 0.0

        except Exception as e:
            logger.debug(f"[大盘] 获取上交所市值失败: {e}")
            return 0.0

    def _get_szse_market_cap(self) -> float:
        """获取深交所总市值（亿元）"""
        try:
            import requests
            import json
            from datetime import timedelta
            today = datetime.now()
            date_str = today.strftime('%Y-%m-%d')

            try:
                # 直接调用深交所API
                url = f"https://www.szse.cn/api/report/ShowReport/data"
                params = {
                    "SHOWTYPE": "json",
                    # 页面属性是1803_after不知道是啥含义，返回的字段就又不一样了
                    "CATALOGID": "1803_sczm",
                    "TABKEY": "tab1",
                    "txtQueryDate": date_str,
                    "random": str(time.time())
                }
                logger.info("那不成调2次？")

                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://www.szse.cn/market/stock/summary/index.html",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
                }

                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()

                data = response.json()
                
                # 解析响应数据
                if isinstance(data, list) and len(data) > 0:
                    table_data = data[0].get('data', [])
                    # logger.info("调接口拿到的值,",table_data)
                    for row in table_data:
                        # 尝试不同的字段名
                        if row.get('lbmc') == '股票' or row.get('zqlb') == '股票' or row.get('证券类别') == '股票':
                            # 尝试不同的总市值字段名
                            cap_str = row.get('sjzz', '') or row.get('zsz', '') or row.get('总市值', '')
                            if cap_str:
                                # 移除逗号和单位，转换为浮点数
                                cap_str = cap_str.replace(',', '').replace('亿元', '')
                                try:
                                    cap = float(cap_str)
                                    logger.info(f"[大盘] 深交所股票总市值: {cap:.2f}亿 (日期: {date_str})")
                                    return cap
                                except ValueError:
                                    logger.info(f"[大盘] 总市值转换失败: {cap_str}")

            except Exception as e:
                logger.debug(f"[大盘] 深交所API调用失败 (日期: {date_str}): {e}")

            # 直接API调用失败，使用akshare作为兜底方案
            # logger.info("[大盘] 直接API调用失败，使用akshare作为兜底方案")
            # from datetime import timedelta
            # today = datetime.now()

            # # 尝试最近5个交易日
            # for i in range(1):
            #     check_date = today - timedelta(days=i)
            #     date_str = check_date.strftime('%Y%m%d')

            #     try:
            #         df = ak.stock_szse_summary(date=date_str)

            #         if df is not None and not df.empty:
            #             # 查找股票总市值行
            #             cap_row = df[df['证券类别'] == '股票']
            #             if not cap_row.empty:
            #                 cap = float(cap_row.iloc[0].get('总市值', 0) or 0)
            #                 # 单位转换：元 -> 亿元
            #                 if cap > 1e12:  # 如果数值很大，说明单位是元
            #                     cap = cap / 1e8
            #                 logger.debug(f"[大盘] akshare获取深交所股票总市值: {cap:.2f}亿 (日期: {date_str})")
            #                 return cap
            #     except Exception as e:
            #         logger.debug(f"[大盘] akshare调用失败 (日期: {date_str}): {e}")
            #         continue

            return 0.0

        except Exception as e:
            logger.debug(f"[大盘] 获取深交所市值失败: {e}")
            return 0.0

    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []
        today = datetime.now()
        month_str = f"{today.year}年{today.month}月"
        
        # 多维度搜索
        search_queries = [
            f"A股 大盘 复盘 {month_str}",
            f"股市 行情 分析 今日 {month_str}",
            f"A股 市场 热点 板块 {month_str}",
        ]
        
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
            
            for query in search_queries:
                # 使用 search_stock_news 方法，传入"大盘"作为股票名
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name="大盘",
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        
        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        try:
            logger.info("[大盘] 调用大模型生成复盘报告...")
            
            generation_config = {
                'temperature': 0.7,
                'max_output_tokens': 2048,
            }
            
            # 根据 analyzer 使用的 API 类型调用
            if self.analyzer._use_openai:
                # 使用 OpenAI 兼容 API
                review = self.analyzer._call_openai_api(prompt, generation_config)
            else:
                # 使用 Gemini API
                response = self.analyzer._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                review = response.text.strip() if response and response.text else None
            
            if review:
                logger.info(f"[大盘] 复盘报告生成成功，长度: {len(review)} 字符")
                return review
            else:
                logger.warning("[大盘] 大模型返回为空")
                return self._generate_template_review(overview, news)
                
        except Exception as e:
            logger.error(f"[大盘] 大模型生成复盘报告失败: {e}")
            return self._generate_template_review(overview, news)

    def _format_margin_data_for_prompt(self, overview: MarketOverview) -> str:
        """格式化逃顶指标数据用于 AI prompt"""
        if overview.margin_ratio <= 0:
            return "暂无数据"

        warning_text = "【警告】触发逃顶信号！" if overview.is_bull_top_warning else ""
        return f"""- 两市融资余额: {overview.margin_balance:.0f}亿
                - 两市总市值: {overview.total_market_cap:.0f}亿
                - 融资/市值比: {overview.margin_ratio:.2f}% (阈值: 3.5%)
                {warning_text}"""

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        # 板块信息
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])

        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"
        
        prompt = f"""你是一位专业的A股市场分析师，请根据以下数据生成一份简洁的大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_text if indices_text else "暂无指数数据（接口异常）"}

## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元
- 北向资金: {overview.north_flow:+.2f} 亿元

## 牛市逃顶指标
{self._format_margin_data_for_prompt(overview)}

## 板块表现
领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}

## 市场新闻
{news_text if news_text else "暂无相关新闻"}

{"注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。" if not indices_text else ""}

---

# 输出格式模板（请严格按此格式输出）

## 📊 {overview.date} 大盘复盘

### 一、市场总结
（2-3句话概括今日市场整体表现，包括指数涨跌、成交量变化）

### 二、指数点评
（分析上证、深证、创业板等各指数走势特点）

### 三、资金动向
（解读成交额和北向资金流向的含义）

### 四、逃顶指标分析
（分析融资余额/总市值比值，判断市场是否过热，是否触发逃顶警告）

### 五、热点解读
（分析领涨领跌板块背后的逻辑和驱动因素）

### 六、后市展望
（结合当前走势和新闻，给出明日市场预判）

### 七、风险提示
（需要关注的风险点，特别是如果逃顶指标异常需要重点提示）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
        return prompt

    def _format_bull_top_indicator(self, overview: MarketOverview) -> str:
        """
        格式化牛市逃顶指标展示

        Args:
            overview: 市场概览数据

        Returns:
            格式化后的逃顶指标文本
        """
        if overview.margin_ratio <= 0:
            return "暂无数据（融资余额或总市值数据获取失败）"

        # 状态判断
        if overview.is_bull_top_warning:
            status = "🔴 **警告：触发逃顶信号！**"
            tip = "融资余额占比超过 3.5%，市场过热，建议谨慎操作"
        elif overview.margin_ratio > 3.0:
            status = "🟡 **关注：接近警戒线**"
            tip = "融资余额占比接近 3.5%，市场情绪偏热"
        elif overview.margin_ratio > 2.5:
            status = "🟢 **正常：市场情绪适中**"
            tip = "融资余额占比处于合理区间"
        else:
            status = "🟢 **安全：市场情绪偏冷**"
            tip = "融资余额占比较低，市场相对安全"

        return f"""| 指标 | 数值 |
|------|------|
| 两市融资余额 | {overview.margin_balance:.0f}亿 |
| 两市总市值 | {overview.total_market_cap:.0f}亿 |
| 融资/市值比 | **{overview.margin_ratio:.2f}%** |
| 逃顶阈值 | 3.5% |

{status}

> 💡 {tip}"""

    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        
        # 判断市场走势
        sh_index = next((idx for idx in overview.indices if idx.code == '000001'), None)
        if sh_index:
            if sh_index.change_pct > 1:
                market_mood = "强势上涨"
            elif sh_index.change_pct > 0:
                market_mood = "小幅上涨"
            elif sh_index.change_pct > -1:
                market_mood = "小幅下跌"
            else:
                market_mood = "明显下跌"
        else:
            market_mood = "震荡整理"
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_text = "、".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = "、".join([s['name'] for s in overview.bottom_sectors[:3]])
        
        report = f"""## 📊 {overview.date} 大盘复盘

### 一、市场总结
今日A股市场整体呈现**{market_mood}**态势。

### 二、主要指数
{indices_text}

### 三、涨跌统计
| 指标 | 数值 |
|------|------|
| 上涨家数 | {overview.up_count} |
| 下跌家数 | {overview.down_count} |
| 涨停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 两市成交额 | {overview.total_amount:.0f}亿 |
| 北向资金 | {overview.north_flow:+.2f}亿 |

### 四、牛市逃顶指标
{self._format_bull_top_indicator(overview)}

### 五、板块表现
- **领涨**: {top_text}
- **领跌**: {bottom_text}

### 六、风险提示
市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
        return report
    
    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程
        
        Returns:
            复盘报告文本
        """
        logger.info("========== 开始大盘复盘分析 ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview()
        
        # 2. 搜索市场新闻
        news = self.search_market_news()
        
        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        
        logger.info("========== 大盘复盘分析完成 ==========")
        
        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
