#include<RcppArmadillo.h>
using namespace arma;
using namespace Rcpp;

//[[Rcpp::depends(RcppArmadillo)]]

int n, m;
double a, c, lambda1c, lambdac;
uvec actset;
vec y;
cube k;


inline double knorm(vec v, mat k_slice_m) {
	return sqrt(dot(k_slice_m*v, v));
}

inline void actset_calcub1n(mat alpha, unsigned gamma, vec rho, double cc) {
	for (int i = 0; i < m; i++) {
		if (knorm(alpha.col(i) + gamma*rho, k.slice(i)) < (gamma*cc)) {
			actset(i) = 0;
		}
		else {
			actset(i) = 1;
		}
	}
}

inline void actset_calcuen(mat alpha, unsigned gamma, vec rho) {
	for (int i = 0; i < m; i++) {
		if (knorm(alpha.col(i) + gamma*rho, k.slice(i)) <= gamma*lambda1c) {
			actset(i) = 0;
		}
		else {
			actset(i) = 1;
		}
	}
}

inline vec proxb1n(vec v, double cc, mat k_slice_m) {
	double norm_value = knorm(v, k_slice_m);
	if (norm_value <= cc) {
		return zeros(n);
	}
	else {
		return ((norm_value - cc) / norm_value)*v;
	}
}

inline vec proxen(vec v, unsigned gamma, mat k_slice_m) {
	double norm_value = knorm(v, k_slice_m);
	if (norm_value <= gamma*lambda1c) {
		return zeros(n);
	}
	else {
		return ((norm_value - gamma*lambda1c) / (1 + gamma*lambdac) / norm_value)*v;
	}
}


inline vec gradlogib1n(vec rho, mat alpha, double b, unsigned gamma, double cc) {
	vec term1, term2, term3 = zeros(n);
	actset_calcub1n(alpha, gamma, rho, cc);
	term1 = y%log(y%rho / (1 - y%rho));
	term2 = (b + gamma*sum(rho))*ones(n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			term3 += k.slice(i)*proxb1n(alpha.col(i) + gamma*rho, cc*gamma, k.slice(i));
		}
	}
	return term1 + term2 + term3;
}

inline vec gradlmumb1n(vec rho, mat alpha, double b, unsigned gamma, double cc) {
	vec term1, term2, term3 = zeros(n);
	actset_calcub1n(alpha, gamma, rho, cc);
	term1 = (a - c) / (c + 1)*y - a / (c + 1)*y%pow(y%rho, -1 / (a + 1));
	term2 = (b + gamma*sum(rho))*ones(n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			term3 += k.slice(i)*proxb1n(alpha.col(i) + gamma*rho, cc*gamma, k.slice(i));
		}
	}
	return term1 + term2 + term3;
}


inline vec gradsvmb1n(vec rho, mat alpha, double b, vec ksi, vec eta, unsigned gamma, double cc) {
	vec term1, term2, term3 = zeros(n), yrho = y%rho;
	actset_calcub1n(alpha, gamma, rho, cc);
	uvec ksi0 = find(ksi > 0), eta0 = find(eta > 0);
	int nksi = ksi0.n_rows;
	term1 = -y;
	term1.elem(ksi0) += y.elem(ksi0) % (ksi.elem(ksi0) - gamma*(ones(nksi) - yrho.elem(ksi0)));
	term1.elem(eta0) -= y.elem(eta0) % (eta.elem(eta0) - gamma*yrho.elem(eta0));
	term2 = (b + gamma*sum(rho))*ones(n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			term3 += k.slice(i)*proxb1n(alpha.col(i) + gamma*rho, cc*gamma, k.slice(i));
		}
	}
	return term1 + term2 + term3;
}


inline vec gradlogien(vec rho, mat alpha, double b, unsigned gamma) {
	vec term1, term2, term3 = zeros(n);
	actset_calcuen(alpha, gamma, rho);
	term1 = y%log(y%rho / (1 - y%rho));
	term2 = (b + gamma*sum(rho))*ones(n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			term3 += k.slice(i)*proxen(alpha.col(i) + gamma*rho, gamma, k.slice(i));
		}
	}
	return term1 + term2 + term3;

}

inline vec gradlmumen(vec rho, mat alpha, double b, unsigned gamma) {
	vec term1, term2, term3 = zeros(n);
	actset_calcuen(alpha, gamma, rho);
	term1 = (a - c) / (c + 1)*y - a / (c + 1)*y%pow(y%rho, -1 / (a + 1));
	term2 = (b + gamma*sum(rho))*ones(n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			term3 += k.slice(i)*proxen(alpha.col(i) + gamma*rho, gamma, k.slice(i));
		}
	}
	return term1 + term2 + term3;
}


inline vec gradsvmen(vec rho, mat alpha, double b, vec ksi, vec eta, unsigned gamma) {
	vec term1, term2, term3 = zeros(n), yrho = y%rho;
	actset_calcuen(alpha, gamma, rho);
	uvec ksi0 = find(ksi > 0), eta0 = find(eta > 0);
	int nksi = ksi0.n_rows;
	term1 = -y;
	term1.elem(ksi0) += y.elem(ksi0) % (ksi.elem(ksi0) - gamma*(ones(nksi) - yrho.elem(ksi0)));
	term1.elem(eta0) -= y.elem(eta0) % (eta.elem(eta0) - gamma*yrho.elem(eta0));
	term2 = (b + gamma*sum(rho))*ones(n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			term3 += k.slice(i)*proxen(alpha.col(i) + gamma*rho, gamma, k.slice(i));
		}
	}
	return term1 + term2 + term3;
}


inline mat hessianlogib1n(vec rho, mat alpha, double b, unsigned gamma, double cc) {
	double temp;
	vec q = zeros(m);
	mat term1, term2, term3 = zeros(n, n), v = zeros(n, m);
	term1 = diagmat(y / (rho % (1 - y%rho)));
	term2 = gamma*ones(n, n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			temp = knorm(alpha.col(i) + gamma*rho, k.slice(i));
			q(i) = gamma*cc / temp;
			v.col(i) = (alpha.col(i) + gamma*rho) / temp;
			term3 += (1 - q(i))*k.slice(i) + q(i)*k.slice(i)*v.col(i)*trans(v.col(i))*k.slice(i);
		}
	}
	return term1 + term2 + gamma*term3 + eye(n, n)*1e-8;
}

inline mat hessianlmumb1n(vec rho, mat alpha, double b, unsigned gamma, double cc) {
	double temp, qm, gc = gamma*cc;
	vec gr = gamma*rho, vm, agr;
	mat term1, term2, term3 = zeros(n, n);
	term1 = diagmat(a / (a + 1) / (c + 1)*pow(y%rho, -(a + 2) / (a + 1)));
	term2 = gamma*ones(n, n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			agr = alpha.col(i) + gr;
			temp = 1 / knorm(agr, k.slice(i));
			qm = gc*temp;
			vm = agr*temp;
			term3 += (1 - qm)*k.slice(i) + qm*k.slice(i)*vm*trans(vm)*k.slice(i);
		}
	}
	return term1 + term2 + gamma*term3 + eye(n, n)*1e-8;
}


inline mat hessiansvmb1n(vec rho, mat alpha, double b, vec ksi, vec eta, unsigned gamma, double cc) {
	double temp, qm, gc = gamma*cc;
	vec gr = gamma*rho, vm, agr;
	mat term1 = zeros(n, n), term2, term3 = zeros(n, n);
	uvec ksi0 = find(ksi > 0), eta0 = find(eta > 0);
	int nksi = ksi0.n_rows, neta = eta0.n_rows;
	term1.submat(ksi0, ksi0) = gamma*eye(nksi, nksi);
	term1.submat(eta0, eta0) += gamma*eye(neta, neta);
	term2 = gamma*ones(n, n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			agr = alpha.col(i) + gr;
			temp = 1 / knorm(agr, k.slice(i));
			qm = gc*temp;
			vm = agr*temp;
			term3 += (1 - qm)*k.slice(i) + qm*k.slice(i)*vm*trans(vm)*k.slice(i);
		}
	}
	return term1 + term2 + gamma*term3 + eye(n, n)*1e-8;
}


inline mat hessianlogien(vec rho, mat alpha, unsigned gamma) {
	double temp, gl1c = gamma*lambda1c, glc = 1 + gamma*lambdac, qm;
	vec gr = gamma*rho, vm, agr;
	mat term1, term2, term3 = zeros(n, n);
	term1 = diagmat(y / (rho % (1 - y%rho)));
	term2 = gamma*ones(n, n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			agr = alpha.col(i) + gr;
			temp = 1 / knorm(agr, k.slice(i));
			qm = gl1c*temp;
			vm = agr*temp;
			term3 += (1 - qm)*k.slice(i) + qm*k.slice(i)*vm*trans(vm)*k.slice(i);
		}
	}
	return term1 + term2 + gamma / glc*term3 + eye(n, n)*1e-8;
}


inline mat hessianlmumen(vec rho, mat alpha, double b, unsigned gamma) {
	double temp, gl1c = gamma*lambda1c, glc = 1 + gamma*lambdac, qm;
	vec gr = gamma*rho, vm, agr;
	mat term1, term2, term3 = zeros(n, n);
	term1 = diagmat(a / (a + 1) / (c + 1)*pow(y%rho, -(a + 2) / (a + 1)));
	term2 = gamma*ones(n, n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			agr = alpha.col(i) + gr;
			temp = 1 / knorm(agr, k.slice(i));
			qm = gl1c*temp;
			vm = temp*agr;
			term3 += (1 - qm)*k.slice(i) + qm*k.slice(i)*vm*trans(vm)*k.slice(i);
		}
	}
	return term1 + term2 + gamma / glc*term3 + eye(n, n)*1e-8;
}


inline mat hessiansvmen(vec rho, mat alpha, double b, vec ksi, vec eta, unsigned gamma) {
	double temp, gl1c = gamma*lambda1c, glc = 1 + gamma*lambdac, qm;
	vec gr = gamma*rho, vm, agr;
	mat term1 = zeros(n, n), term2, term3 = zeros(n, n);
	uvec ksi0 = find(ksi > 0), eta0 = find(eta > 0);
	int nksi = ksi0.n_rows, neta = eta0.n_rows;
	term1.submat(ksi0, ksi0) = gamma*eye(nksi, nksi);
	term1.submat(eta0, eta0) += gamma*eye(neta, neta);
	term2 = gamma*ones(n, n);
	for (int i = 0; i < m; i++) {
		if (actset(i)) {
			agr = alpha.col(i) + gr;
			temp = 1 / knorm(agr, k.slice(i));
			qm = gl1c*temp;
			vm = temp*agr;
			term3 += (1 - qm)*k.slice(i) + qm*k.slice(i)*vm*trans(vm)*k.slice(i);
		}
	}
	return term1 + term2 + gamma / glc*term3 + eye(n, n)*1e-8;
}


inline vec newtonlogib1n(vec rho0, mat alpha, double b, unsigned gamma, double cc, int maxiter2, double cri2) {
	int i;
	double ss, st, s1, s2;
	vec rho = rho0, rho_new, g, yd, hg, yrhonew, yrho;
	uvec ydx0, ydd0;
	mat h(n, n);
	for (i = 0; i < maxiter2; i++) {
		Rcout << "inner iter=" << i << "\t";
		g = gradlogib1n(rho, alpha, b, gamma, cc);
		h = hessianlogib1n(rho, alpha, b, gamma, cc);
		hg = -solve(h, g);
		rho_new = rho + hg;
		yrho = y%rho;
		yrhonew = y%rho_new;
		if (any(yrhonew <= 0) || any(yrhonew >= 1)) {
			yd = y%hg;
			ydx0 = find(yd < 0);
			ydd0 = find(yd > 0);
			if (ydx0.n_rows > 0) {
				s1 = min(-yrho.elem(ydx0) / yd.elem(ydx0))*0.99;
			}
			else {
				s1 = 1;
			}
			if (ydd0.n_rows > 0) {
				s2 = min((ones(ydd0.n_rows) - yrho.elem(ydd0)) / yd.elem(ydd0))*0.99;
			}
			else {
				s2 = 1;
			}
			ss = (s1 > s2) ? s2 : s1;
			st = (ss > 1) ? 1 : ss;
			rho_new = rho + st*hg;
		}
		if (norm(rho - rho_new)/norm(rho)<cri2) {
			break;
		}
		rho = rho_new;
	}
	if (i == maxiter2) {
		Rcout << "Does not converge in the inner cycle" << endl;
	}
	return rho;
}

inline vec newtonlmumb1n(vec rho0, mat alpha, double b, unsigned gamma, double cc, int maxiter2, double cri2) {
	int i;
	double ss, st, s1, s2;
	vec rho = rho0, rho_new, g, yd, hg, yrhonew, yrho;
	uvec ydx0, ydd0;
	mat h(n, n);
	for (i = 0; i < maxiter2; i++) {
		Rcout << "inner iter=" << i << "\t";
		g = gradlmumb1n(rho, alpha, b, gamma, cc);
		h = hessianlmumb1n(rho, alpha, b, gamma, cc);
		hg = -solve(h, g);
		rho_new = rho + hg;
		yrho = y%rho;
		yrhonew = y%rho_new;
		if (any(yrhonew <= 0) || any(yrhonew >= 1)) {
			yd = y%hg;
			ydx0 = find(yd < 0);
			ydd0 = find(yd > 0);
			if (ydx0.n_rows > 0) {
				s1 = min(-yrho.elem(ydx0) / yd.elem(ydx0))*0.99;
			}
			else {
				s1 = 1;
			}
			if (ydd0.n_rows > 0) {
				s2 = min((ones(ydd0.n_rows) - yrho.elem(ydd0)) / yd.elem(ydd0))*0.99;
			}
			else {
				s2 = 1;
			}
			ss = (s1 > s2) ? s2 : s1;
			st = (ss > 1) ? 1 : ss;
			rho_new = rho + st*hg;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          st*hg;
		}
		if (norm(rho - rho_new) / norm(rho)<cri2) {
			break;
		}
		rho = rho_new;
	}
	if (i == maxiter2) {
		Rcout << "Does not converge in the inner cycle" << endl;
	}
	return rho;
}

inline vec newtonsvmb1n(vec rho0, mat alpha, double b, vec ksi, vec eta, unsigned gamma, double cc, int maxiter2, double cri2) {
	int i;
	double ss, st, s1, s2;
	vec rho = rho0, rho_new, g, yd, hg, yrhonew, yrho;
	uvec ydx0, ydd0;
	mat h(n, n);
	for (i = 0; i < maxiter2; i++) {
		Rcout << "inner iter=" << i << "\t";
		g = gradsvmb1n(rho, alpha, b, ksi, eta, gamma, cc);
		h = hessiansvmb1n(rho, alpha, b, ksi, eta, gamma, cc);
		hg = -solve(h, g);
		rho_new = rho + hg;
		yrho = y%rho;
		yrhonew = y%rho_new;
		if (any(yrhonew <= 0) || any(yrhonew >= 1)) {
			yd = y%hg;
			ydx0 = find(yd < 0);
			ydd0 = find(yd > 0);
			if (ydx0.n_rows > 0) {
				s1 = min(-yrho.elem(ydx0) / yd.elem(ydx0))*0.99;
			}
			else {
				s1 = 1;
			}
			if (ydd0.n_rows > 0) {
				s2 = min((ones(ydd0.n_rows) - yrho.elem(ydd0)) / yd.elem(ydd0))*0.99;
			}
			else {
				s2 = 1;
			}
			ss = (s1 > s2) ? s2 : s1;
			st = (ss > 1) ? 1 : ss;
			rho_new = rho + st*hg;
		}
		if (norm(rho - rho_new) / norm(rho)<cri2) {
			break;
		}
		rho = rho_new;
	}
	if (i == maxiter2) {
		Rcout << "Does not converge in the inner cycle" << endl;
	}
	return rho;
}


inline vec newtonlogien(vec rho0, mat alpha, double b, unsigned gamma, int maxiter2, double cri2) {
	int i;
	double ss, st, s1, s2;
	vec rho = rho0, rho_new, g, yd, hg, yrhonew, yrho;
	uvec ydx0, ydd0;
	mat h(n, n);
	for (i = 0; i < maxiter2; i++) {
		g = gradlogien(rho, alpha, b, gamma);
		h = hessianlogien(rho, alpha, gamma);
		hg = -solve(h, g);
		rho_new = rho + hg;
		yrho = y%rho;
		yrhonew = y%rho_new;
		if (any(yrhonew <= 0) || any(yrhonew >= 1)) {
			yd = y%hg;
			ydx0 = find(yd < 0);
			ydd0 = find(yd > 0);
			if (ydx0.n_rows > 0) {
				s1 = min(-yrho.elem(ydx0) / yd.elem(ydx0))*0.99;
			}
			else {
				s1 = 1;
			}
			if (ydd0.n_rows > 0) {
				s2 = min((ones(ydd0.n_rows) - yrho.elem(ydd0)) / yd.elem(ydd0))*0.99;
			}
			else {
				s2 = 1;
			}
			ss = (s1 > s2) ? s2 : s1;
			st = (ss > 1) ? 1 : ss;
			rho_new = rho + st*hg;
		}
		if (norm(rho - rho_new) / norm(rho)<cri2) {
			break;
		}
		rho = rho_new;
	}
	if (i == maxiter2) {
		Rcout << "Does not converge in the inner cycle" << endl;
	}
	return rho;
}



inline vec newtonlmumen(vec rho0, mat alpha, double b, unsigned gamma, int maxiter2, double cri2) {
	int i;
	double ss, st, s1, s2;
	vec rho = rho0, rho_new, g, yd, hg, yrhonew, yrho;
	uvec ydx0, ydd0;
	mat h(n, n);
	for (i = 0; i < maxiter2; i++) {
		Rcout << "inner iter=" << i << "\t";
		g = gradlmumen(rho, alpha, b, gamma);
		h = hessianlmumen(rho, alpha, b, gamma);
		hg = -solve(h, g);
		rho_new = rho + hg;
		yrho = y%rho;
		yrhonew = y%rho_new;
		if (any(yrhonew <= 0) || any(yrhonew >= 1)) {
			yd = y%hg;
			ydx0 = find(yd < 0);
			ydd0 = find(yd > 0);
			if (ydx0.n_rows > 0) {
				s1 = min(-yrho.elem(ydx0) / yd.elem(ydx0))*0.99;
			}
			else {
				s1 = 1;
			}
			if (ydd0.n_rows > 0) {
				s2 = min((ones(ydd0.n_rows) - yrho.elem(ydd0)) / yd.elem(ydd0))*0.99;
			}
			else {
				s2 = 1;
			}
			ss = (s1 > s2) ? s2 : s1;
			st = (ss > 1) ? 1 : ss;
			rho_new = rho + st*hg;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          st*hg;
		}
		if (norm(rho - rho_new) / norm(rho)<cri2) {
			break;
		}
		rho = rho_new;
	}
	if (i == maxiter2) {
		Rcout << "Does not converge in the inner cycle" << endl;
	}
	return rho;
}


inline vec newtonsvmen(vec rho0, mat alpha, double b, vec ksi, vec eta, unsigned gamma, int maxiter2, double cri2) {
	int i;
	double ss, st, s1, s2;
	vec rho = rho0, rho_new, g, yd, hg, yrhonew, yrho;
	uvec ydx0, ydd0;
	mat h(n, n);
	for (i = 0; i < maxiter2; i++) {
		Rcout << "inner iter=" << i << "\t";
		g = gradsvmen(rho, alpha, b, ksi, eta, gamma);
		h = hessiansvmen(rho, alpha, b, ksi, eta, gamma);
		hg = -solve(h, g);
		rho_new = rho + hg;
		yrho = y%rho;
		yrhonew = y%rho_new;
		if (any(yrhonew <= 0) || any(yrhonew >= 1)) {
			yd = y%hg;
			ydx0 = find(yd < 0);
			ydd0 = find(yd > 0);
			if (ydx0.n_rows > 0) {
				s1 = min(-yrho.elem(ydx0) / yd.elem(ydx0))*0.99;
			}
			else {
				s1 = 1;
			}
			if (ydd0.n_rows > 0) {
				s2 = min((ones(ydd0.n_rows) - yrho.elem(ydd0)) / yd.elem(ydd0))*0.99;
			}
			else {
				s2 = 1;
			}
			ss = (s1 > s2) ? s2 : s1;
			st = (ss > 1) ? 1 : ss;
			rho_new = rho + st*hg;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          st*hg;
		}
		if (norm(rho - rho_new) / norm(rho)<cri2) {
			break;
		}
		rho = rho_new;
	}
	if (i == maxiter2) {
		Rcout << "Does not converge in the inner cycle" << endl;
	}
	return rho;
}




//[[Rcpp::export]]
List logib1n(vec y0, cube k0, mat alpha, double b, vec rho, double cc, int maxiter1, int maxiter2, double cri1, double cri2) {
	n = y0.n_rows;
	m = alpha.n_cols;
	int i, j;
	unsigned gamma = 2;
	y = y0;
	k = k0;
	mat alpha_new(n, m);
	actset = ones<uvec>(m);
	for (i = 0; i < maxiter1; i++) {
		Rcout << "outer iter=" << i << endl;
		rho = newtonlogib1n(rho, alpha, b, gamma, cc, maxiter2, cri2);
		for (j = 0; j < m; j++) { // j=1,2,...,m
			alpha_new.col(j) = proxb1n(alpha.col(j) + gamma*rho, cc*gamma, k.slice(j));
		}
		b += gamma*sum(rho);

		Rcout << sum(rho) << endl;
		gamma *= 2;

		if ((gamma*fabs(sum(rho))) < cri1) {
			Rcout << "outer" << i << endl;
			break;
		}
		alpha = alpha_new;
	}
	if (i == maxiter1) {
		Rcout << "Does not converge in the outer cycle" << endl;
	}
	List res;
	res["b"] = b;
	res["alpha"] = alpha;
	return res;
}


//[[Rcpp::export]]
List lmumb1n(vec y0, cube k0, double a0, double c0, mat alpha, double b, vec rho, double cc, int maxiter1, int maxiter2, double cri1, double cri2) {
	n = y0.n_rows;
	m = alpha.n_cols;
	a = a0;
	c = c0;
	int i, j;
	unsigned gamma = 10;
	y = y0;
	k = k0;
	mat alpha_new(n, m);
	actset = ones<uvec>(m);
	for (i = 0; i < maxiter1; i++) {
		Rcout << "outer iter=" << i << endl;
		rho = newtonlmumb1n(rho, alpha, b, gamma, cc, maxiter2, cri2);
		for (j = 0; j < m; j++) {
			alpha_new.col(j) = proxb1n(alpha.col(j) + gamma*rho, gamma*cc, k.slice(j));
		}
		b += gamma*sum(rho);

		Rcout << sum(rho) << endl;
		gamma *= 10;
		alpha = alpha_new;
		if ((gamma*fabs(sum(rho))) < cri1) {
			Rcout << "outer" << i << endl;
			break;
		}
	}
	if (i == maxiter1) {
		Rcout << "Does not converge in the outer cycle" << endl;
	}
	List res;
	res["b"] = b;
	res["alpha"] = alpha;
	res["nk"] = size(nonzeros(actset), 0);
	return res;
}


//[[Rcpp::export]]
List svmb1n(vec y0, cube k0, mat alpha, double b, vec ksi, vec eta, vec rho, double cc, int maxiter1, int maxiter2, double cri1, double cri2) {
	n = y0.n_rows;
	m = alpha.n_cols;
	int i, j;
	unsigned gamma = 10;
	y = y0;
	k = k0;
	mat alpha_new(n, m);
	actset = ones<uvec>(m);
	vec yrho;
	for (i = 0; i < maxiter1; i++) {
		Rcout << "outer iter=" << i << endl;
		rho = newtonsvmb1n(rho, alpha, b, ksi, eta, gamma, cc, maxiter2, cri2);
		yrho = y%rho;
		for (j = 0; j < m; j++) { // j=1,2,...,m
			alpha_new.col(j) = proxb1n(alpha.col(j) + gamma*rho, cc*gamma, k.slice(j));
		}
		for (j = 0;j < n;j++) {
			ksi(j) = (ksi(j) - gamma*(1 - yrho(j))>0) ? (ksi(j) - gamma*(1 - yrho(j))) : 0;
			eta(j) = (eta(j) - gamma*yrho(j)>0) ? (eta(j) - gamma*yrho(j)) : 0;
		}
		b += gamma*sum(rho);
		gamma *= 10;
		alpha = alpha_new;
		Rcout << "sumrho" << sum(rho) << endl;
		if ((gamma*fabs(sum(rho))) < cri1) {
			Rcout << "outer" << i << endl;
			break;
		}
	}
	if (i == maxiter1) {
		cout << "Does not converge in the outer cycle" << endl;
	}
	List res;
	res["b"] = b;
	res["alpha"] = alpha;
	res["nk"] = size(nonzeros(actset), 0);
	return res;
}



//[[Rcpp::export]]
List logien(vec y0, cube k0, mat alpha, double b0, vec rho0, double cc, double lambda, int maxiter1, int maxiter2, double cri1, double cri2) {
	n = y0.n_rows;
	m = alpha.n_cols;
	int i, j;
	unsigned gamma = 10;
	double b = b0;
	lambdac = cc*lambda, lambda1c = cc*(1 - lambda);
	vec rho = rho0;
	y = y0;
	k = k0;
	mat alpha_new(n, m);
	actset = ones<uvec>(m);
	for (i = 0; i < maxiter1; i++) {
		rho = newtonlogien(rho, alpha, b, gamma, maxiter2, cri2);
		for (j = 0; j < m; j++) {
			alpha_new.col(j) = proxen(alpha.col(j) + gamma*rho, gamma, k.slice(j));
		}
		b += gamma*sum(rho);
		if ((gamma*fabs(sum(rho))) < cri1) {
			break;
		}
		gamma *= 10;
		alpha = alpha_new;
	}
	if (i == maxiter1) {
		Rcout << "Does not converge in the outer cycle" << endl;
	}
	List res;
	res["b"] = b;
	res["alpha"] = alpha;
	return res;
}



//[[Rcpp::export]]
List lmumen(vec y0, cube k0, double a0, double c0, mat alpha, double b, vec rho, double cc, double lambda, int maxiter1, int maxiter2, double cri1, double cri2) {
	n = y0.n_rows;
	m = alpha.n_cols;
	a = a0;
	c = c0;
	int i, j;
	unsigned gamma = 10;
	lambdac = cc*lambda, lambda1c = cc*(1 - lambda);
	y = y0;
	k = k0;
	mat alpha_new(n, m);
	actset = ones<uvec>(m);
	for (i = 0; i < maxiter1; i++) {
		Rcout << "outer iter=" << i << endl;
		rho = newtonlmumen(rho, alpha, b, gamma, maxiter2, cri2);
		for (j = 0; j < m; j++) { // j=1,2,...,m
			alpha_new.col(j) = proxen(alpha.col(j) + gamma*rho, gamma, k.slice(j));
		}
		b += gamma*sum(rho);

		Rcout << sum(rho) << endl;
		gamma *= 10;
		alpha = alpha_new;
		if ((gamma*fabs(sum(rho))) < cri1) {
			Rcout << "outer" << i << endl;
			break;
		}
	}
	if (i == maxiter1) {
		Rcout << "Does not converge in the outer cycle" << endl;
	}
	List res;
	res["b"] = b;
	res["alpha"] = alpha;
	res["nk"] = size(nonzeros(actset), 0);
	return res;
}


//[[Rcpp::export]]
List svmen(vec y0, cube k0, mat alpha, double b, vec ksi, vec eta, vec rho, double cc, double lambda, int maxiter1, int maxiter2, double cri1, double cri2) {
	n = y0.n_rows;
	m = alpha.n_cols;
	int i, j;
	unsigned gamma = 10;
	lambdac = cc*lambda, lambda1c = cc*(1 - lambda);
	vec yrho;
	y = y0;
	k = k0;
	mat alpha_new(n, m);
	actset = ones<uvec>(m);
	for (i = 0; i < maxiter1; i++) {
		Rcout << "outer iter=" << i << endl;
		rho = newtonsvmen(rho, alpha, b, ksi, eta, gamma, maxiter2, cri2);
		yrho = y%rho;
		for (j = 0; j < m; j++) { // j=1,2,...,m
			alpha_new.col(j) = proxen(alpha.col(j) + gamma*rho, gamma, k.slice(j));
		}
		for (j = 0;j < n;j++) {
			ksi(j) = (ksi(j) - gamma*(1 - yrho(j))>0) ? (ksi(j) - gamma*(1 - yrho(j))) : 0;
			eta(j) = (eta(j) - gamma*yrho(j)>0) ? (eta(j) - gamma*yrho(j)) : 0;
		}
		b += gamma*sum(rho);

		Rcout << sum(rho) << endl;
		gamma *= 10;
		alpha = alpha_new;
		if ((gamma*fabs(sum(rho))) < cri1) {
			Rcout << "outer" << i << endl;
			break;
		}
	}
	if (i == maxiter1) {
		Rcout << "Does not converge in the outer cycle" << endl;
	}
	List res;
	res["alpha"] = alpha;
	res["b"] = b;
	return res;
}


//[[Rcpp::export]]
vec predict(mat alpha, double b, cube k) {
	mat kk;
	int mm = k.n_cols, p = k.n_slices;
	vec y = zeros(mm);
	for (int j = 0; j < mm; j++) {
		double temp = 0;
		for (int i = 0; i < p; i++) {
			kk = k.slice(i);
			temp += dot(kk.col(j), alpha.col(i));
		}
		y(j) = temp + b;
	}
	return y;
}
