import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

// Carregado sob demanda (React.lazy nos gráficos): o Plotly completo é pesado e só o detalhe o usa.
const Plot = createPlotlyComponent(Plotly);
export default Plot;
