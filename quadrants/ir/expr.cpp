#include "quadrants/ir/expr.h"

#include "quadrants/ir/frontend_ir.h"
#include "quadrants/ir/ir.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

void Expr::set_dbg_info(const DebugInfo &dbg_info) {
  expr->dbg_info = dbg_info;
}

const std::string &Expr::get_tb() const {
  return expr->get_tb();
}

DataType Expr::get_ret_type() const {
  return expr->ret_type;
}

DataType Expr::get_rvalue_type() const {
  if (auto argload = cast<ArgLoadExpression>()) {
    if (argload->is_ptr) {
      return argload->ret_type.ptr_removed();
    }
    return argload->ret_type;
  }
  if (auto id = cast<IdExpression>()) {
    return id->ret_type.ptr_removed();
  }
  if (auto index_expr = cast<IndexExpression>()) {
    return index_expr->ret_type.ptr_removed();
  }
  if (auto unary = cast<UnaryOpExpression>()) {
    if (unary->type == UnaryOpType::frexp) {
      return unary->ret_type.ptr_removed();
    }
    return unary->ret_type;
  }
  return expr->ret_type;
}

void Expr::type_check(const CompileConfig *config) {
  expr->type_check(config);
}

Expr cast(const Expr &input, DataType dt) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::cast_value, input, dt);
}

Expr bit_cast(const Expr &input, DataType dt) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::cast_bits, input, dt);
}

namespace {

// Bottom-up clone of every BinaryOp / UnaryOp / TernaryOp expression reachable from `cur`, tagging the fresh Binary /
// Unary nodes `precise`. Non-walked kinds (loads, constants, qd.func calls, ndarray accesses, ...) carry no `precise`
// field and are passed through by reference - aliasing them is safe. TernaryOp nodes are cloned structurally so the
// walk can recurse into their branches, but the TernaryOp itself does not carry a `precise` flag (the only ternary
// today is `select`, a control-flow-shaped conditional move, not FP arithmetic; see also the matching comment in expr.h
// and the `precise` fields in frontend_ir.h / statements.h).
Expr clone_and_tag_precise(const Expr &cur) {
  if (auto bin = cur.cast<BinaryOpExpression>()) {
    Expr new_lhs = clone_and_tag_precise(bin->lhs);
    Expr new_rhs = clone_and_tag_precise(bin->rhs);
    Expr out = Expr::make<BinaryOpExpression>(bin->type, new_lhs, new_rhs);
    auto new_bin = out.cast<BinaryOpExpression>();
    new_bin->precise = true;
    new_bin->dbg_info = bin->dbg_info;
    new_bin->attributes = bin->attributes;
    new_bin->ret_type = bin->ret_type;
    return out;
  }
  if (auto un = cur.cast<UnaryOpExpression>()) {
    Expr new_operand = clone_and_tag_precise(un->operand);
    Expr out = un->is_cast() ? Expr::make<UnaryOpExpression>(un->type, new_operand, un->cast_type, un->dbg_info)
                             : Expr::make<UnaryOpExpression>(un->type, new_operand, un->dbg_info);
    auto new_un = out.cast<UnaryOpExpression>();
    new_un->precise = true;
    new_un->attributes = un->attributes;
    new_un->ret_type = un->ret_type;
    return out;
  }
  if (auto tri = cur.cast<TernaryOpExpression>()) {
    Expr new_op1 = clone_and_tag_precise(tri->op1);
    Expr new_op2 = clone_and_tag_precise(tri->op2);
    Expr new_op3 = clone_and_tag_precise(tri->op3);
    Expr out = Expr::make<TernaryOpExpression>(tri->type, new_op1, new_op2, new_op3);
    auto new_tri = out.cast<TernaryOpExpression>();
    new_tri->dbg_info = tri->dbg_info;
    new_tri->attributes = tri->attributes;
    new_tri->ret_type = tri->ret_type;
    return out;
  }
  return cur;
}

}  // namespace

Expr precise(const Expr &input) {
  // Return a fresh Expression subtree with every reachable BinaryOp and UnaryOp tagged `precise`. The user's original
  // subtree is untouched: no in-place mutation, so aliasing a subexpression
  // (`ab = a + b; x = qd.precise(ab); y = ab * 2`) does not retroactively tag the other alias. Non-walked kinds (loads,
  // constants, qd.func calls, ndarray accesses, ...) are passed through by reference; they carry no `precise` field, so
  // sharing them is safe. See expr.h for the full canonical contract.
  return clone_and_tag_precise(input);
}

Expr &Expr::operator=(const Expr &o) {
  set(o);
  return *this;
}

SNode *Expr::snode() const {
  QD_ASSERT_INFO(is<FieldExpression>(), "Cannot get snode of non-field expressions.");
  return cast<FieldExpression>()->snode;
}

void Expr::set_adjoint(const Expr &o) {
  this->cast<FieldExpression>()->adjoint.set(o);
}

void Expr::set_dual(const Expr &o) {
  this->cast<FieldExpression>()->dual.set(o);
}

void Expr::set_adjoint_checkbit(const Expr &o) {
  this->cast<FieldExpression>()->adjoint_checkbit.set(o);
}

Expr::Expr(uint1 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::u1, x);
}

Expr::Expr(int16 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::i16, x);
}

Expr::Expr(int32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::i32, x);
}

Expr::Expr(int64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::i64, x);
}

Expr::Expr(float32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::f32, x);
}

Expr::Expr(float64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(PrimitiveType::f64, x);
}

Expr::Expr(const Identifier &id) : Expr() {
  expr = std::make_shared<IdExpression>(id);
}

Expr expr_rand(DataType dt) {
  return Expr::make<RandExpression>(dt);
}

Expr assume_range(const Expr &expr, const Expr &base, int low, int high, const DebugInfo &dbg_info) {
  return Expr::make<RangeAssumptionExpression>(expr, base, low, high, dbg_info);
}

Expr loop_unique(const Expr &input, const std::vector<SNode *> &covers, const DebugInfo &dbg_info) {
  return Expr::make<LoopUniqueExpression>(input, covers, dbg_info);
}

Expr expr_field(Expr id_expr, DataType dt) {
  QD_ASSERT(id_expr.is<IdExpression>());
  auto ret = Expr(std::make_shared<FieldExpression>(dt, id_expr.cast<IdExpression>()->id));
  return ret;
}

Expr expr_matrix_field(const std::vector<Expr> &fields, const std::vector<int> &element_shape) {
  return Expr::make<MatrixFieldExpression>(fields, element_shape);
}

}  // namespace quadrants::lang
