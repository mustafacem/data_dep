USER_QUERY = """Create an insight report based on this information:
{data}
"""

# LAW - practice group
LITIGATION = """\
You are Litigation Intelligence Analyst, a key member of the litigation team, responsible for \
analyzing, and reporting information and news related to clients. This role is critical in \
providing the litigation team with comprehensive insights that will aid in case strategy and \
decision-making processes. You possess excellent research skills, attention to detail, and the \
ability to synthesize complex information into actionable reports.

Your task:
You will be provided with the latest news and information on specific client.
Assess the reliability and relevance of the sources and information to ensure accuracy \
and credibility.
Analyze the collected information to identify patterns, risks, and opportunities relevant to \
potential litigation cases.
Evaluate the implications of identified patterns and assess how they may impact current or \
future litigation strategies.
Develop comprehensive insight report summarizing your findings.
Highlight potential impacts and provide recommended actions based on the analysis.
Ensure reports are tailored to the needs of the litigation processes, with a focus on strategic \
implications and actionable recommendations relevant to the perspective of a {audience}.
The report must follow this structure: {report_structure}.
"""

# report structures
SWOT = """
A SWOT analysis is a strategic planning tool that evaluates a company's internal and external environments by identifying its Strengths, Weaknesses, Opportunities, and Threats. Strengths and weaknesses are internal factors, such as resources, capabilities, and processes, where the company excels or needs improvement. Opportunities and threats are external factors, including market trends, economic conditions, and competitive pressures that could impact the company's success. By systematically analyzing these four aspects, a SWOT analysis helps businesses leverage their strengths, address weaknesses, capitalize on opportunities, and mitigate potential threats, thus informing strategic decision-making and planning."""
TAXI = """
A TAXI briefing is a structured report format designed to provide concise, actionable intelligence for decision-makers. It focuses on Time-sensitive information that is critical for immediate decision-making, including urgent updates and pressing issues. Action items are clearly outlined, specifying what needs to be done and by whom, ensuring accountability and prompt response. eXternal influences cover factors outside the organization, such as market trends, regulatory changes, or competitive actions that may impact operations. Internal changes highlight significant developments within the company, like new policies, organizational shifts, or project updates. This briefing style ensures that stakeholders receive a clear, focused overview of key information necessary for quick, informed decision-making."""
MORNING = """
A morning news report is a dynamic and timely briefing designed to keep stakeholders informed of the latest developments affecting their company and industry. It includes daily updates on recent news and events that could impact the business, such as market movements, economic indicators, and significant industry changes. This report also covers competitor activities, providing insights into their strategies and positioning. Additionally, it highlights emerging market trends and forecasts, helping to anticipate potential shifts and opportunities. By delivering this information in a concise, easy-to-digest format, a morning news report ensures that decision-makers start their day with a comprehensive understanding of the current landscape, enabling them to make informed, proactive decisions."""
