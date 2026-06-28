package test

import "core:math"

RigidBody :: struct {
    force: Vector3,
    velocity: Vector3,
    torque: Vector3,
    angularVelocity: Vector3,
    bounciness: f32,
    friction: f32,
    isStatic: bool,
    mass: f32,
    invMass: f32,
    isMovingBySupport: bool
}

AddForceAtPoint :: proc(model: ^Model, force: Vector3, contactPoint: Vector3) {
    model.rigidBody.force += force

    r := contactPoint - model.translation
    torque := Vector3CrossProduct(r, force)
    model.rigidBody.torque += torque
}

ApplyPhysics :: proc(models: []Model, deltaTime: f32) {
    for &model in models {
        if model.rigidBody.isStatic do continue

        ApplyGravity(&model, models, deltaTime)
        IntegrateLinearForce(&model, deltaTime)
        IntegrateTorque(&model, deltaTime)
    }
}

ApplyGravity :: proc(model: ^Model, models: []Model, deltaTime: f32) {
    isGrounded := false

    for &other in models {
        if &other == model do continue

        model.translation.y -= GROUND_PROBE_DIST
        probeResult := GetCollisionResult(model, &other)
        model.translation.y += GROUND_PROBE_DIST

        if probeResult.hit {
            if HasBoxCollider(model) {
                ApplyFriction(model, other)
                ApplyStabilization(model, probeResult.normal)

                if HasBoxCollider(&other)  {
                    overhang := GetOverhang(model, other)
                    surfaceAngle := math.acos(abs(Vector3DotProduct(probeResult.normal, WORLD_UP)))

                    if overhang.x <= 1e-6 && overhang.y <= 1e-6 && surfaceAngle < SLIDE_ANGLE_THRESHOLD {
                        isGrounded = true
                    }

                    if (overhang.x > 1e-6 || overhang.y > 1e-6) && probeResult.normal.y <= -0.5 {
                        ApplyTipping(model, probeResult.contactPoint, overhang)
                    }
                }
            }

            break
        }
    }

    if HasSphereCollider(model) {
        ApplyRolling(model)
    }

    if !isGrounded {
        model.rigidBody.velocity += GRAVITY * deltaTime
    }

    GetOverhang :: proc(model: ^Model, other: Model) -> Vector2 {
        otherAxes := GetAxesFromRotationMatrix(other.rotationMatrix)
        center := model.translation - other.translation
        projX := abs(Vector3DotProduct(center, otherAxes[0]))
        projZ := abs(Vector3DotProduct(center, otherAxes[2]))
        halfX := other.collider.(BoxCollider).size.x * other.scale
        halfZ := other.collider.(BoxCollider).size.z * other.scale
        return { projX - halfX, projZ - halfZ }
    }

    ApplyFriction :: proc(model: ^Model, other: Model) {
        avgFriction := (model.rigidBody.friction + other.rigidBody.friction) * 0.5
        model.rigidBody.force.x *= avgFriction
        model.rigidBody.force.z *= avgFriction
        model.rigidBody.torque.y *= avgFriction
    }

    ApplyRolling :: proc(model: ^Model) {
        speed := Vector3Length(model.rigidBody.velocity)
        if speed > MIN_VELOCITY_THRESHOLD && !model.rigidBody.isMovingBySupport {
            axis := Vector3CrossProduct(WORLD_UP, model.rigidBody.velocity / speed)
            radius := model.collider.(SphereCollider).radius * model.scale
            model.rigidBody.angularVelocity = axis * (speed / radius)
        } else {
            model.rigidBody.angularVelocity *= 0.1
        }   
    }

    ApplyStabilization :: proc(model: ^Model, normal: Vector3) {
        closestUp  := FindClosestUpAxis(model.rotationMatrix, normal)
        correction := Vector3CrossProduct(closestUp, normal)
        model.rigidBody.torque += correction * STABILIZATION_STRENGTH
    }

    ApplyTipping :: proc(model: ^Model, contactPoint: Vector3, overhang: Vector2) {
        rContact := model.translation - contactPoint
        gravTorque := Vector3CrossProduct(rContact, GRAVITY)
        if overhang.x > 1e-6 do gravTorque.x -= overhang.x
        if overhang.y > 1e-6 do gravTorque.z -= overhang.y
        model.rigidBody.torque += Vector3Normalize(gravTorque) * model.rigidBody.mass * TIPPING_STRENGTH
    }
}

IntegrateLinearForce :: proc(model: ^Model, deltaTime: f32) {
    model.rigidBody.velocity += model.rigidBody.force * model.rigidBody.invMass * deltaTime
    model.rigidBody.force = {}

    drag: f32
    if HasBoxCollider(model) { drag = LINEAR_DRAG } else { drag = SPHERE_LINEAR_DRAG }
    model.rigidBody.velocity *= drag
    
    if Vector3Length(model.rigidBody.velocity) > MIN_VELOCITY_THRESHOLD {
        model.translation += model.rigidBody.velocity * deltaTime
    }
}

IntegrateTorque :: proc(model: ^Model, deltaTime: f32) {
    model.rigidBody.angularVelocity += model.rigidBody.torque * model.rigidBody.invMass * deltaTime
    model.rigidBody.torque = {}

    drag: f32
    if HasBoxCollider(model) { drag = ANGULAR_DRAG } else { drag = SPHERE_ANGULAR_DRAG }
    model.rigidBody.angularVelocity *= drag

    avlength  := Vector3Length(model.rigidBody.angularVelocity)
    if avlength > MIN_ANGULAR_VELOCITY_THRESHOLD {
        axis := model.rigidBody.angularVelocity / avlength
        angle := avlength * deltaTime
        delta := MakeRotationMatrixAxisAngle(axis, angle)
        model.rotationMatrix = Mat4Mul(delta, model.rotationMatrix)
    }
}

FindClosestUpAxis :: proc(rotationMatrix: Matrix4x4, referenceUp: Vector3) -> Vector3 {
    axes := GetAxesFromRotationMatrix(rotationMatrix)

    closestAxis := axes[0]
    closestDot := abs(Vector3DotProduct(axes[0], referenceUp))

    for i in 1..<3 {
        d := abs(Vector3DotProduct(axes[i], referenceUp))
        if d > closestDot {
            closestDot  = d
            closestAxis = axes[i]
        }
    }

    if Vector3DotProduct(closestAxis, referenceUp) < 0 {
        closestAxis = -closestAxis
    }

    return closestAxis
}